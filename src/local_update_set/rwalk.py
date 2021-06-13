#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import DatasetSplit
import copy 
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader, self.fisherloader = self.train_val_test(
            dataset, list(idxs))
        #self.device = 'cuda' if args.gpu else 'cpu'
        self.device = 'cuda:%d' % args.gpu if args.gpu is not None else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.ewc_lambda = args.ewc_lambda

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        #TODO: check fisher loader
        fisherloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.fisher_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader, fisherloader

    def update_weights(self, model, fisher, global_round):
        # Set mode to train model
        fixed_model = copy.deepcopy(model) 
        fixed_params = {n:p for n,p in fixed_model.named_parameters()}
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                
                fixed_log_probs = fixed_model(images)
                fixed_loss = self.criterion(fixed_log_probs, labels)
                delta_loss = loss-fixed_loss

                #EWC loss
                #information = {}
                if not global_round == 0: #TODO first step! -> Not using fisher info
                #    for n, p in model.named_parameters():
                #        information[n] = nn.functional.relu(delta_loss/(0.5*fisher[n]*(p-fixed_params[n])**2+1e-10))
                    reg_loss = 0 
                    for n, p in model.named_parameters():
                        #reg_loss += ((fisher[n]+information[n])*((p-fixed_params[n])**2)).sum()
                        reg_loss += (fisher[n]*((p-fixed_params[n])**2)).sum()
                    loss += self.ewc_lambda * reg_loss * 0.5
                loss.backward()
                optimizer.step()
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        fisher = self.update_fisher(model, fixed_model, fisher)
        
        return model.state_dict(), fisher, sum(epoch_loss) / len(epoch_loss)

    def compute_diag_fisher(self, model, fixed_model):
       
        #Define optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        #initialization of fisher info
        diag_fisher = {}
        score_information = {} 
        for n, p in model.named_parameters():
            diag_fisher[n] = p.clone().detach().fill_(0)
            score_information[n] = p.clone().detach().fill_(0)
        
        fixed_params = {n:p for n,p in fixed_model.named_parameters()}
        previous_params = fixed_params
        previous_model = fixed_model
        #diagonal fisher matrix
        for i, (images, labels) in enumerate(self.fisherloader):
            optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            log_probs = model(images)
            
            #batch_size = images.shape[0]
            #TODO true label? estimated label?  
            loss = self.criterion(log_probs, labels)
            loss.backward()
            
            #optimizer.step()
            fixed_log_probs = previous_model(images)
            fixed_loss = self.criterion(fixed_log_probs, labels)
            fixed_loss.backward()
            previous_params = {n:p for n,p in previous_model.named_parameters()}
            delta_loss = loss-fixed_loss
            for n, p in model.named_parameters():
                if p.grad is not None:
                    diag_fisher[n] += (p.grad.detach()**2 / len(self.fisherloader))
            for n, p in model.named_parameters():
                eps = 1e-5
                if p.grad is not None:
                    #batch_score_info = -delta_loss/(0.5*diag_fisher[n]*(p-fixed_params[n])**2+eps)
                    #batch_score_info =-1*(p.grad -fixed_params[n].grad)/(0.5*diag_fisher[n]*(p-fixed_params[n])**2+1e-1)
                    batch_score_info =-p.grad*(p-previous_params[n])/(0.5*diag_fisher[n]*(p-fixed_params[n])**2+eps)
                    #batch_score_info =-p.grad*(p-previous_params[n])/(0.5*(p-fixed_params[n])**2+eps)
                    score_information[n] += nn.functional.relu(batch_score_info.detach()/len(self.fisherloader))
            #previous_params = {n:p.detach() for n,p in model.named_parameters()}
            previous_model = model
        return diag_fisher, score_information
    
   
    def update_fisher(self, model, fixed_model, fisher=None):
        diag_fisher, score_information = self.compute_diag_fisher(model, fixed_model)
        
        if fisher is None:
            print('first step for fisher information')
            return diag_fisher
        
        elif self.args.fisher_update_type=='own':
            return diag_fisher
        
        elif self.args.fisher_update_type == 'summation':
            for n, p in model.named_parameters():
                fisher[n] += diag_fisher[n] + score_information[n]
            return fisher 
        
        elif self.args.fisher_update_type=='gamma':
            for n, p in model.named_parameters():
                fisher[n] = (1-self.args.gamma)*diag_fisher[n] + self.args.gamma*fisher[n] + score_information[n]#TODO gamma
                fisher[n] = fisher[n].detach()
            return fisher 
 
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

