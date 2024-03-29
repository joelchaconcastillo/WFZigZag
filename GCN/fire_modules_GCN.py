import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs

def Laplacian_link(supports, link_len):
        node_num, _ = supports.shape
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        return torch.stack(support_set, dim=0)

def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return np.stack(cheb_polynomials, axis = 0)

def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

class CNN(nn.Module):
   def __init__(self, dim_input, dim_out):
      super(CNN, self).__init__()
      self.dim_out = dim_out

      self.features = nn.Sequential(
          nn.Conv2d(dim_input, int(dim_out/2), kernel_size=3, stride=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(int(dim_out/2), dim_out, kernel_size=3, stride=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )
      self.maxpool = nn.AdaptiveMaxPool2d((1,1))

   def forward(self, graph):
      feature = self.features(graph)
      feature = self.maxpool(feature)
      feature = feature.view(-1, self.dim_out) #B, dim_out
      return feature

class SpatioTemporalGCN(nn.Module):
    def __init__(self, dim_in, hidden_dim, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos = 0):
        super(SpatioTemporalGCN, self).__init__()

        subhidden_num = 3  #number of sub-hidden layers (adaptive adj matrix, time matrix, spatial matrix)
        self.link_len = link_len
        self.hidden_dim = hidden_dim
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(hidden_dim/subhidden_num)))
        self.weights_pool_adj = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(hidden_dim/subhidden_num)))
        if layer_pos == 0:
           self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(hidden_dim / subhidden_num)))
        else:
           self.weights_window = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in/subhidden_num), int(hidden_dim / subhidden_num)))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))
        self.T = nn.Parameter(torch.FloatTensor(window_len))
#        self.cnn = CNN(num_persis_diagrams, int(hidden_dim/subhidden_num))
        self.supports_adj = supports_adj


    def forward(self, x, x_window, node_embeddings, ZPI):
        '''
           x: B, N, F
           node_num :  N, E
        '''
        (batch_size, lag, node_num, dim) = x_window.shape
        #S1: Graph construction, a suggestion is to pre-process graph, however since wildfire requires ~1TB for pre-processing graph we create it from fly
        #S2: Laplacian construction

        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        #S3: Laplacianlink
        for k in range(2, self.link_len):
            support_set.append(torch.mm(supports, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        #S4: spatial graph convolution
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #N, link_len, dim_in, hidden_dim/2 : on E
        bias = torch.matmul(node_embeddings, self.bias_pool) #N, hidden_dim : on E
        x_g = torch.einsum("knm,bmc->bknc", supports, x) #B, link_len, N, dim_in : on N
        x_g = x_g.permute(0, 2, 1, 3) #B, N, link_len, dim_in  : on 
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) #B, N, hidden_dim/2
        x_gconv = F.normalize(x_gconv, dim=-1)


        weights_adj = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool_adj) #N, link_len, dim_in, hidden_dim/2 : on E
        x_a = torch.einsum("knn,bmc->bknc", self.supports_adj, x) #B, link_len, N, dim_in : on N
        x_a = x_a.permute(0, 2, 1, 3) #B, N, link_len, dim_in  : on 
        x_aconv = torch.einsum('bnki,nkio->bno', x_a, weights_adj) #B, N, hidden_dim/2
        x_aconv = F.normalize(x_aconv, dim=-1)


        #S5: temporal feature transformation
        weights_window = torch.einsum('nd,dio->nio', node_embeddings, self.weights_window)  #N, dim_in, hidden_dim/2 : on E
        x_w = torch.einsum('btni,nio->btno', x_window, weights_window)  #B, T, N, hidden_dim/2: on D
        x_w = x_w.permute(0, 2, 3, 1)  #B, N, hidden_dim/2, T 
        x_wconv = torch.matmul(x_w, self.T)  #B, N, hidden_dim/2: on T
        x_wconv = F.normalize(x_wconv, dim=-1)
#        #S6: Transform graph information to [hidden_dim/2, hidden_dim/2] 
#        convZPI =  F.normalize(self.cnn(ZPI), dim=-1) ## B, H/2
        x_tgconv =  x_gconv #torch.einsum('bno,bo->bno',x_gconv, convZPI) #B, N, H/2
        x_taconv =  x_aconv #torch.einsum('bno,bo->bno',x_aconv, convZPI) #B, N, H/2
        x_twconv =  x_wconv #torch.einsum('bno,bo->bno', x_wconv, convZPI) #B, N, H/2

#        #S7: combination operation
        x_gwconv = torch.cat([x_tgconv, x_taconv, x_twconv], dim = -1) + bias  # + bias_adj #B, N, hidden_dim
        return x_gwconv

class GCN_GRU_Cell(nn.Module):
    def __init__(self, node_num, dim_in, hidden_dim, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos):
        super(GCN_GRU_Cell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.gate = SpatioTemporalGCN(dim_in+self.hidden_dim, 2*hidden_dim, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos)
        self.update = SpatioTemporalGCN(dim_in+self.hidden_dim, hidden_dim, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, layer_pos)

    def forward(self, x, state, x_full, node_embeddings, ZPI):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        h_current = state.to(x.device)
        input_and_state = torch.cat((x, h_current), dim=-1) #x + state
        z_r = torch.sigmoid(self.gate(input_and_state, x_full, node_embeddings, ZPI))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r*h_current), dim=-1)
        n = torch.tanh(self.update(candidate, x_full, node_embeddings, ZPI))
        h_next = (1.0-z)*n + z*h_current
        return h_next

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class GCN_GRU(nn.Module):
    def __init__(self, node_num, dim_in, hidden_dim, link_len, embed_dim, num_layers=1, window_len = 10, num_persis_diagrams=2, supports_adj=None, return_all_layers=False):
        super(GCN_GRU, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.return_all_layers = return_all_layers 
        self.node_num = node_num
        self.input_dim = dim_in
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_len = window_len
        self.cell_list = nn.ModuleList()
        for i in range(0, num_layers):
            cur_input_dim = dim_in if i == 0 else hidden_dim
            self.cell_list.append(GCN_GRU_Cell(node_num, cur_input_dim, hidden_dim, window_len, link_len, embed_dim, num_persis_diagrams, supports_adj, i))

    def forward(self, x, ZPI, node_embeddings, hidden_state = None):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        (batch_size, seq_len, input_dim, n) = x.shape
        hidden_state = self.init_hidden(batch_size)
        layer_output_list = []
        last_state_list = []
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                state = self.cell_list[layer_idx](cur_layer_input[:, t, :, :], state, cur_layer_input, node_embeddings, ZPI)
                output_inner.append(state)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([state])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden_state(batch_size))
        return init_states



class GCN(nn.Module):
   def __init__(self, args):
      super().__init__()
      input_dim = len(args.static_features) + len(args.dynamic_features)
      if args.clc == 'vec':
          input_dim += 10
      self.num_nodes = int(args.patch_width)*int(args.patch_height)
      self.patch_width = args.patch_width
      self.patch_height = args.patch_height
      self.hidden_dim = args.rnn_units
      self.input_dim = args.input_dim
      self.output_dim = args.output_dim
      self.num_layers = args.num_layers
      self.embed_dim = args.embed_dim
      self.window_len = args.window_len
      self.link_len = args.link_len
      self.horizon = args.horizon
      num_persis_diagrams = 2*len(args.scaleParameter)
      dropout=0.5

      self.ln1 = torch.nn.LayerNorm(self.input_dim)
      self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        #predictor
#      self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(self.num_nodes, self.hidden_dim), bias=True)

        # fully-connected part
#      self.ln2 = torch.nn.LayerNorm(self.hidden_dim)
#      self.conv1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1))
      self.fc1 = nn.Linear(self.num_nodes*self.hidden_dim, 2)
      #self.fc1 = nn.Linear(int(self.patch_width//2)*int(self.patch_height//2)*self.hidden_dim, 2 * self.hidden_dim)
      self.drop1 = nn.Dropout(dropout)
###
#      self.fc2 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
#      self.drop2 = nn.Dropout(dropout)
###
#      self.fc3 = nn.Linear(self.hidden_dim, 2)
#      self.fc3 = nn.Linear(int(self.hidden_dim/2), 2)

      self.adj = np.identity((self.num_nodes))
      nextR = [0, 1, 1, 1, 0, -1, -1, -1]  #displacement by rows
      nextC = [-1, -1, 0, 1, 1, 1, 0, -1]  #displacement by cols
      ##note that we are considering self-loops!!
      ##create graph with 8 neightbours..
      middlePixelR = self.patch_height/2
      middlePixelC = self.patch_width/2
      for i in range(self.patch_height):
#          if i < middlePixelR-1 or i > middlePixelR+1:
#              continue
          for j in range(self.patch_width):
#              if j < middlePixelC-1 or j > middlePixelC+1:
#                 continue
              id_node = i*self.patch_width + j
              for k in range(len(nextR)):
                  nr, nc = i+nextR[k], j+nextC[k]
                  if nr<0 or nr>=self.patch_height or nc <0 or nc>=self.patch_width:
                      continue
                  id_node_next = nr*self.patch_width + nc
                  self.adj[id_node, id_node_next] +=1
                  self.adj[id_node_next, id_node] +=1
#      self.supports_adj = torch.from_numpy(cheb_polynomial(scaled_Laplacian(self.adj), self.link_len)).float()
      support = torch.from_numpy(self.adj).float().to(args.device)
      support /= torch.sum(support, dim=1) #normalize by rows..
      supports_adj = Laplacian_link(support, self.link_len)

      self.encoder = GCN_GRU(self.num_nodes, self.input_dim, self.hidden_dim, self.link_len, self.embed_dim, self.num_layers, self.window_len, num_persis_diagrams, supports_adj)


   def forward(self, x: torch.Tensor, ZPI: torch.Tensor):
      '''
         x :     batch, time, features, nodes (width x height pixels)
         graph:  batch, time, nodes, nodes
         target:  batch, prediction
      '''
      x = x.permute(0, 1, 3, 2).float() ## B, T, N, D
      (B,T,N,D) = x.shape
      x = self.ln1(x)
      x, _ = self.encoder(x, ZPI.float(), self.node_embeddings) #B, T, N, hidden_dim
      x = x[0][:, -1:, :, :] #B, 1, N, hidden_dim
#      #CNN based predictor
#      x = self.end_conv((x)) #B, T*C, N, 1
#      x = x.squeeze(-1).reshape(-1, self.output_dim)
#      return torch.nn.functional.log_softmax(x, dim=-1)
#      print(x[0,...],"<===")
#      x = self.ln2(x)
      x = x.squeeze(1).permute(0,2,1) # B, hidden_dim, N
      x = x.reshape(B, self.hidden_dim, self.patch_width, self.patch_height) # B, N,  H

      #x = F.max_pool2d(F.relu(self.conv1(x)), 2) #B, hidden, 12, 12

      # fully-connected
      x = torch.flatten(x, 1) ##B, hidden*12*12 (4608)
      x = (self.drop1(self.fc1(x)))  ##B, 64

#      x = F.relu(self.drop2(self.fc2(x)))  ##B, 32
      #x = F.relu(self.fc2(x))  ##B, 32
#      x = self.fc3(x) ##B, 2
      return torch.nn.functional.log_softmax(x, dim=1)
