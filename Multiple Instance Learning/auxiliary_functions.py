# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:03:24 2023

@author: AmayaGS

"""

import os
import random
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset

class Accuracy_Logger(object):
    """Accuracy logger for storing both individual results and summary statistics."""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
        self.all_labels = []
        self.all_predictions = []

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
        self.all_labels.append(Y)
        self.all_predictions.append(Y_hat)

    def log_batch(self, Y_hat, Y):
        Y_hat = Y_hat.astype(int)
        Y = Y.astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
        self.all_labels.extend(Y.tolist())
        self.all_predictions.extend(Y_hat.tolist())

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

    def get_accuracy(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        return float(correct) / count if count != 0 else None

    def get_all_predictions_and_labels(self):
        return self.all_predictions, self.all_labels

# class Accuracy_Logger(object):

#     """Accuracy logger"""
#     def __init__(self, n_classes):
#         #super(Accuracy_Logger, self).__init__()
#         self.n_classes = n_classes
#         self.initialize()

#     def initialize(self):
#         self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

#     def log(self, Y_hat, Y):
#         Y_hat = int(Y_hat)
#         Y = int(Y)
#         self.data[Y]["count"] += 1
#         self.data[Y]["correct"] += (Y_hat == Y)

#     def log_batch(self, Y_hat, Y):
#         Y_hat = np.array(Y_hat).astype(int)
#         Y = np.array(Y).astype(int)
#         for label_class in np.unique(Y):
#             cls_mask = Y == label_class
#             self.data[label_class]["count"] += cls_mask.sum()
#             self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

#     def get_summary(self, c):
#         count = self.data[c]["count"]
#         correct = self.data[c]["correct"]

#         if count == 0:
#             acc = None
#         else:
#             acc = float(correct) / count

#         return acc, correct, count

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True