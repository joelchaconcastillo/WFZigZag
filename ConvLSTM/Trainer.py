import torch
import math
import os
import time
import copy
import numpy as np
import scipy.sparse as sp
from logger import get_logger
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn import metrics

#from metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader1, test_loader2,
                 scaler, args, lr_scheduler=None, path=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader1 = test_loader1
        self.test_loader2 = test_loader2
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.number_minibatches = int(args.batch_size/args.minbatch_size)
        self.minbatch_size = args.minbatch_size
        self.batch_size = args.batch_size
        self.train_per_epoch = len(train_loader)
        self.start_epoch = 0
        ############TODO: read an existent model
        if path is not None:
           self.load_checkpoint(path)
        #####################3
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('log dir: {}'.format(args.log_dir))
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        self.logger.info("Argument: %r", args)
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        countBatches = 0.
        preds = np.array([])
        targets = np.array([])
        probs = np.array([])
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(val_dataloader):
                output = self.model(data)

                preds = np.concatenate((preds, torch.argmax(output, dim=1).cpu().numpy()))
                targets= np.concatenate((targets, label.cpu().numpy()))
                probs = np.concatenate((probs, (torch.exp(output)[:, 1]).cpu().numpy()))

                loss = self.loss(output, label)

                loss = loss/self.batch_size
                
                if ( (batch_idx+1)%self.number_minibatches == 0 ) or ((batch_idx+1) == len(val_dataloader)):
                    countBatches +=1.
                if not torch.isnan(loss):
                   total_val_loss += loss.item()

        tn, fp, fn, tp = metrics.confusion_matrix(targets, preds).ravel()
        auc = roc_auc_score(targets, probs)
        aucpr = average_precision_score(targets, probs)
        summary = classification_report(targets, preds, digits=3, output_dict=True)['1.0']
        summary['AUC']=auc
        summary['AUCPR']=aucpr
        summary['TP']=tp
        summary['FP']=fp
        summary['TN']=tn
        summary['FN']=fn
        self.logger.info("\n metrics validation: {} \n".format(summary))

        val_loss = total_val_loss / countBatches
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_loss_batch = 0
        countBatches = 0
        self.optimizer.zero_grad()
        for batch_idx, (data, label) in enumerate(self.train_loader):

            output = self.model(data)

            loss = self.loss(output, label)
            loss = loss/self.batch_size
            total_loss_batch += loss.item()
           # self.logger.info('{}...'.format(loss.item()))
            loss.backward()
            if ( (batch_idx+1)%self.number_minibatches == 0 ) or ((batch_idx+1) == len(self.train_loader)):
               # add max grad clipping
      #         if self.args.grad_norm:
      #            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
               self.optimizer.step()
               self.optimizer.zero_grad()
               total_loss += total_loss_batch

               #log information
#               if (int((batch_idx+1)/self.number_minibatches)) % self.args.log_step == 0:
               self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_idx, self.train_per_epoch, total_loss_batch))
               total_loss_batch = 0
               countBatches +=1.
        train_epoch_loss = total_loss/countBatches
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} '.format(epoch, train_epoch_loss))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs + 1): # 总 epochs
            epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            self.logger.info('\nEpoch time elapsed: {}\n'.format(time.time() - epoch_time))
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e7:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)

            # apply the best model to test dataset
            # test
            self.model.load_state_dict(best_model)
            # self.val_epoch(self.args.epochs, self.test_loader)
            self.test(self.model, self.args, self.test_loader1, self.scaler, self.logger)
            self.test(self.model, self.args, self.test_loader2, self.scaler, self.logger)


        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader1, self.scaler, self.logger)
        self.test(self.model, self.args, self.test_loader2, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'epoch': self.epoch,
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
    def load_checkpoint(self, path): ##TODO
        check_point = torch.load(path)
        self.args = check_point['config']
        self.model.load_state_dict(check_point['state_dict'])
        self.optimizer.load_state_dict(check_point['optimizer'])
        self.start_epoch = check_point['epoch']
        model.to(self.args.devices)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        preds = np.array([])
        targets = np.array([])
        probs = np.array([])

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader):
                output = model(data)
                preds = np.concatenate((preds, torch.argmax(output, dim=1).cpu().numpy()))
                targets= np.concatenate((targets, label.cpu().numpy()))
                probs = np.concatenate((probs, (torch.exp(output)[:, 1]).cpu().numpy()))
        np.save('./{}_true.npy'.format(args.dataset), targets)
        np.save('./{}_pred.npy'.format(args.dataset), preds)

        tn, fp, fn, tp = metrics.confusion_matrix(targets, preds).ravel()
        auc = roc_auc_score(targets, probs)
        aucpr = average_precision_score(targets, probs)
        summary = classification_report(targets, preds, digits=3, output_dict=True)['1.0']
        summary['AUC']=auc
        summary['AUCPR']=aucpr
        summary['TP']=tp
        summary['FP']=fp
        summary['TN']=tn
        summary['FN']=fn
        logger.info("\n Testing metrics {} \n".format(summary))

