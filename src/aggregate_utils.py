#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0.
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights(w, fisher=None):
    #if w is None:
    #    return w
    """
    Returns the average of the weights.
    """
    w_avg = {}
    for key in w[0].keys():
        w_avg[key] = 0.
        for i in range(0, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def normalize(w):
    w_sum = {}
    for key in w[0].keys():
        w_sum[key] = 0.
        for i in range(0, len(w)):
            w_sum[key] += w[i][key]
        for i in range(0, len(w)):
            w[i][key] /= w_sum[key]
    return w

def average_weights_with_fisher(w, local_fisher):
    local_fisher = normalize(local_fisher)
    w_avg = {}
    for key in w[0].keys():
        w_avg[key] = 0.
        for i in range(0, len(w)):
            w_avg[key] += w[i][key]*local_fisher[i][key]
    return w_avg



