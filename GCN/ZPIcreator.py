from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import warnings
import json
import scipy.sparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import zigzag.zigzagtools as zzt
import zigzag.ZZgraph as zzgraph
from scipy.spatial.distance import squareform
import scipy.sparse as sp
import dionysus as d
import matplotlib.pyplot as plt
import time
from ripser import ripser
import scipy.sparse
import sys
from persim import plot_diagrams, PersImage
path = os.getcwd()

class zigzagTDA:

   def __init__(self, alpha, scaleParameter, maxDimHoles, sizeWindow, sizeBorder):
      self.alpha = alpha
      self.NVertices = NVertices
      self.scaleParameter = scaleParameter
      self.maxDimHoles = maxDimHoles
      self.sizeWindow = sizeWindow
      self.NVertices = (2*sizeBorder+1)**2
      self.sizeBorder=sizeBorder
   #X: T, N, F
   def zigzag_persistence_diagrams(self, x, graphL2):
       GraphsNetX = []
       for t in range(self.sizeWindow): 
#         graphL2[graphL2==0] = 1e-5  ##weakly connected, we want similar or equally edges :)
##         tmp_max = np.max(graphL2)
##         graphL2 /= tmp_max  ##normalize  matrix
##        graphL2[graphL2>self.alpha]=0 ##cut off note: probably this is not required
#         G = nx.from_numpy_matrix(graphL2)
#         edges = sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1)) ##sort edges increasinly by weights 
#         nConnections = 500 ##just take the top connections
#         graphL2[:,:] = 0 ##reset
#         cont = 0
#         for u,v, w in edges:
#             graphL2[u,v]=float(w['weight'])
#             graphL2[v,u]=float(w['weight'])
#             cont +=1
#             if cont == nConnections:
#                 break

         GraphsNetX.append(graphL2)
#       scipy.sparse.save_npz(prefix_path+"_graph", scipy.sparse.csc_matrix(np.concatenate(GraphsNetX, axis=1)))
       start_time = time.time()
      # Building unions and computing distance matrices
#       print("Building unions and computing distance matrices...")  # Beginning
       MDisGUnions = []
       for i in range(0, self.sizeWindow - 1):
           # --- To concatenate graphs
           MDisAux = np.zeros((2 * self.NVertices, 2 * self.NVertices))
           A = GraphsNetX[i] #nx.adjacency_matrix(GraphsNetX[i]).todense()
           B = GraphsNetX[i+1] #nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
           # ----- Version Original (2)
           C = (A + B) / 2
           A[A == 0] = 10.1
           A[range(self.NVertices), range(self.NVertices)] = 0
           B[B == 0] = 10.1
           B[range(self.NVertices), range(self.NVertices)] = 0
           MDisAux[0:self.NVertices, 0:self.NVertices] = A
           C[C == 0] = 10.1
           C[range(self.NVertices), range(self.NVertices)] = 0
           MDisAux[self.NVertices:(2 * self.NVertices), self.NVertices:(2 * self.NVertices)] = B
           MDisAux[0:self.NVertices, self.NVertices:(2 * self.NVertices)] = C
           MDisAux[self.NVertices:(2 * self.NVertices), 0:self.NVertices] = C.transpose()
           # Distance in condensed form
           pDisAux = squareform(MDisAux)
           # --- To save unions and distances
           MDisGUnions.append(pDisAux)  # To save distance matrix
#       print("  --- End unions...")  # Ending
       
       # To perform Ripser computations
#       print("Computing Vietoris-Rips complexes...")  # Beginning
   
       GVRips = []
       for jj in range(self.sizeWindow - 1):
           ripsAux = d.fill_rips(MDisGUnions[jj], self.maxDimHoles, self.scaleParameter)
           GVRips.append(ripsAux)
#       print("  --- End Vietoris-Rips computation")  # Ending
   
       # Shifting filtrations...
#       print("Shifting filtrations...")  # Beginning
       GVRips_shift = []
       GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
       for kk in range(1, self.sizeWindow - 1):
           shiftAux = zzt.shift_filtration(GVRips[kk], self.NVertices * kk)
           GVRips_shift.append(shiftAux)
#       print("  --- End shifting...")  # Ending
   
       # To Combine complexes
#       print("Combining complexes...")  # Beginning
       completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
       for uu in range(2, self.sizeWindow - 1):
           completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
#       print("  --- End combining")  # Ending
   
       # To compute the time intervals of simplices
#       print("Determining time intervals...")  # Beginning
       time_intervals = zzt.build_zigzag_times(completeGVRips, self.NVertices, self.sizeWindow)
#       print("  --- End time")  # Beginning
   
       # To compute Zigzag persistence
#       print("Computing Zigzag homology...")  # Beginning
       G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
#       print("  --- End Zigzag")  # Beginning
   
       # To show persistence intervals
       window_ZPD = []
       # Personalized plot
       for vv, dgm in enumerate(G_dgms):
#           print("Dimension:", vv)
           if (vv < 2):
               matBarcode = np.zeros((len(dgm), 2))
               k = 0
               for p in dgm:
                   matBarcode[k, 0] = p.birth
                   matBarcode[k, 1] = p.death
                   k = k + 1
               matBarcode = matBarcode / 2
               window_ZPD.append(matBarcode)
   
       # Timing
       #print("TIME: " + str((time.time() - start_time)) + " Seg ---  " + str((time.time() - start_time) / 60) + " Min ---  " + str((time.time() - start_time) / (60 * 60)) + " Hr ")
   
       return window_ZPD
   
   # Zigzag persistence image
   def zigzag_persistence_images(self, dgms, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
       if len(dgms) < dimensional: #validation
           return np.zeros(resolution)
       PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
       xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
       x = np.linspace(xm, xM, resolution[0])
       y = np.linspace(ym, yM, resolution[1])
       X, Y = np.meshgrid(x, y)
       Zfinal = np.zeros(X.shape)
       X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]
       # Compute zigzag persistence image
       P0, P1 = np.reshape(dgms[int(dimensional)][:, 0], [1, 1, -1]), np.reshape(dgms[int(dimensional)][:, 1], [1, 1, -1])
       weight = np.abs(P1 - P0)
       distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)
   
       if return_raw:
           lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
           lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
       else:
           weight = weight ** power
           Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)
   
       output = [lw, lsum] if return_raw else Zfinal
       if normalization:
           if np.max(output)-np.min(output) == 0:
               return output
           norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
       else:
           norm_output = output

       return norm_output
#####################################################################
dynamic_features = [
    '1 km 16 days NDVI',
#    '1 km 16 days EVI',
#    'ET_500m',
    'LST_Day_1km',
    'LST_Night_1km',
#    'Fpar_500m',
#    'Lai_500m',
#    'era5_max_u10',
#    'era5_max_v10',
    'era5_max_d2m',
    'era5_max_t2m',
    'era5_max_sp',
    'era5_max_tp',
#    'era5_min_u10',
#    'era5_min_v10',
#    'era5_min_d2m',
#    'era5_min_t2m',
#    'era5_min_sp',
#    'era5_min_tp',
#    'era5_avg_u10',
#    'era5_avg_v10',
#    'era5_avg_d2m',
#    'era5_avg_t2m',
#    'era5_avg_sp',
#    'era5_avg_tp',
#    'smian',
    'sminx',
#    'fwi',
#    'era5_max_wind_u10',
#    'era5_max_wind_v10',
    'era5_max_wind_speed',
#    'era5_max_wind_direction',
#    'era5_max_rh',
    'era5_min_rh',
#    'era5_avg_rh',
]


static_features = [
 'dem_mean',
# 'aspect_mean',
 'slope_mean',
# 'roughness_mean',
 'roads_distance',
 'waterway_distance',
 'population_density',
]
clc = 'vec'
access_mode = 'spatiotemporal'
nan_fill = -1.0 
dataset_root = '/home/joel.chacon/tmp/datasets_grl'
#####TDA parameters
maxDimHoles = 1
window = 10
alpha = 1
scaleParameter =  1.0
sizeBorder = 5#12
NVertices = (2*sizeBorder+1)**2

#data = np.random.rand(window, NVertices, 25)

###We can ge more plots from here...
###zzgraph.plotting(data, NVertices, alpha, scaleParameter, maxDimHoles, window)
### for each sample....
#ZZ = zigzagTDA(alpha=alpha, NVertices=NVertices, scaleParameter=scaleParameter, maxDimHoles=maxDimHoles, sizeWindow=window)

#sample = np.zeros((sizeWindow, NVertices, numberFeatures))
#for t in range(sizeWindow):
#   X = np.concatenate((dynamic[t], static, clc), axis=0) ##F, W, H
#   X = X[:,12-sizeBorder:13+sizeBorder,12-sizeBorder:13+sizeBorder]
#   X = X.reshape(numberFeatures, -1) # F, N
#   sample[t] = X.transpose(1,0) #N, F
#zigzag_PD = ZZ.zigzag_persistence_diagrams(x = sample)
#zigzag_PI_H0 = ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 0)
#zigzag_PI_H1 = ZZ.zigzag_persistence_images(zigzag_PD, dimensional = 1)
#ZPI = [zigzag_PI_H0, zigzag_PI_H1]
