import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
import torch.nn.init as init

import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.data
from dgl.utils import expand_as_pair
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv

__all__ = ['SimpleGNN']

class SimpleGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        '''
        Simple GNN model with multiple layers
        args:
            in_features: number of input features (input dimension)
            hidden_features: list of hidden features
            out_features: number of output features (num_classes)
        '''
        super().__init__()
        # create layers
        self.backbone = nn.ModuleList()
        hidden_features = [in_features] + hidden_features
        for idx in range(1, len(hidden_features)):
            self.backbone.append(GraphConv(hidden_features[idx-1], hidden_features[idx], allow_zero_in_degree=True))
        self.cls_head = GraphConv(hidden_features[-1], out_features, allow_zero_in_degree=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, blocks, x, output_labels = None):
        #print(self.net['backbone'])
        for idx, layer in enumerate(self.backbone):
            x = layer(blocks[idx], x)
            x = self.relu(x)
        x = self.cls_head(blocks[-1], x)
        x = self.softmax(x)
        if self.training == True:
            assert (output_labels is not None)
            x = self.criterion(x, output_labels)
        return x
        