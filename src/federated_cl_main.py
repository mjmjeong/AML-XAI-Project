#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, exp_details, test_inference
from build_method import build_method
from aggregate_utils import average_weights

import random
import numpy as np
import wandb
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Wandb Initialization
    if args.wandb:
        wandb.init(project=args.dataset,entity='amlxai', name=args.desc, notes='../logs')
        wandb.config.update(args)

    
    #if args.gpu_id:
    
    #    torch.cuda.set_device(args.gpu_id)
    #device = 'cuda' if args.gpu else 'cpu'
    device = 'cuda:%d' % args.gpu
    print(device)
    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    #TODO cudnn?
    
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    #with torch.autograd.set_detect_anomaly(True):
#ILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=200,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    if args.wandb:
        wandb.watch(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # define aggregate function and local function
    LocalUpdate, GlobalUpdate = build_method(args)
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    global_fisher = None
    global_scores = None
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_fishers = []
        local_scores = []
        local_fishers_scores = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, fisher, score, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), fisher=copy.deepcopy(global_fisher), 
                score=copy.deepcopy(global_scores),
                global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_fishers.append(copy.deepcopy(fisher))
            local_scores.append(copy.deepcopy(score))
        # update global weights
        
        local_fishers_scores = copy.deepcopy(local_scores)
        for i in range(len(local_fishers)):
            for k, v in local_fishers[i].items(): 
                local_fishers_scores[i][k] = local_fishers[i][k] +local_scores[i][k]
        global_fisher = average_weights(local_fishers)
        global_scores = average_weights(local_scores)
        global_fishers_scores = average_weights(local_fishers_scores)
        global_weights = GlobalUpdate(local_weights, local_fishers_scores)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

            # Test inference after completion of training
            test_acc, test_loss = test_inference(args, global_model, test_dataset)

            print(f' \n Results after {args.epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
            if args.wandb:
                wandb.log({
                'Avg Train Acc' : 100 * train_accuracy[-1],
                'Test Acc' : 100 * test_acc
                })

            # Saving the objects train_loss and train_accuracy:
            file_name = 'save/objects/{}_{}_{}_C{}_iid{}_E{}_B{}_{}_{}_lambda{}_progress.txt'. \
                format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                       args.local_ep, args.local_bs, args.local_update, args.global_update, args.ewc_lambda)

            with open(file_name, 'a') as f:
                print('epoch {}, test_loss {}, train_accuracy {}'.format((epoch+1), test_loss, test_acc), file=f)

            print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = 'save/objects/{}_{}_{}_C{}_iid{}_E{}_B{}_{}_{}_lambda{}.txt'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs,args.local_update, args.global_update,args.ewc_lambda)

    with open(file_name, 'a') as f:
        print('test_loss {}, train_accuracy {}'.format(test_loss, test_acc), file=f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        # PLOTTING (optional)
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('Agg')
    
        # Plot Loss curve
        # plt.figure()
        # plt.title('Training Loss vs Communication rounds')
        # plt.plot(range(len(train_loss)), train_loss, color='r')
        # plt.ylabel('Training loss')
        # plt.xlabel('Communication Rounds')
        # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
        #             format(args.dataset, args.model, args.epochs, args.frac,
        #                    args.iid, args.local_ep, args.local_bs))
        #
        # # Plot Average Accuracy vs Communication rounds
        # plt.figure()
        # plt.title('Average Accuracy vs Communication rounds')
        # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        # plt.ylabel('Average Accuracy')
        # plt.xlabel('Communication Rounds')
        # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
        #             format(args.dataset, args.model, args.epochs, args.frac,
        #                    args.iid, args.local_ep, args.local_bs))
