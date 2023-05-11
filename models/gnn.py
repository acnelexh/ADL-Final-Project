import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
import torch.nn.init as init
from balanced_loss import Loss

import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.data
from dgl.utils import expand_as_pair
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv, SAGEConv, GATv2Conv
from .layer import GraphConvPlus

__all__ = ['SimpleGNN']

class SimpleGNN(nn.Module):
    def __init__(self, args):
        '''
        Simple GNN model with multiple layers
        args:
            in_features: number of input features (input dimension)
            hidden_features: list of hidden features
            out_features: number of output features (num_classes)
        '''
        super().__init__()
        hidden_features = args.hidden_dim
        if args.few_shot == True:
            self.few_shot = True
            in_features = args.input_dim + args.label_embed_dim
        else:
            self.few_shot = False
            in_features = args.input_dim
            
        self.class_balance_loss = args.class_balance_loss
        out_features = args.num_classes
        embedding_dim = args.label_embed_dim
        # create layers
        self.backbone = nn.ModuleList()
        hidden_features = [hidden_features] if type(hidden_features) == int else hidden_features
        hidden_features = [in_features] + hidden_features
        # create backbone
        for idx in range(1, len(hidden_features)):
            if args.edge_weight == False:
                print('Using GraphConv')
                self.backbone.append(GraphConv(hidden_features[idx-1], hidden_features[idx], allow_zero_in_degree=True))
            else:
                print('Using GraphConvPlus')
                self.backbone.append(GraphConvPlus(hidden_features[idx-1], hidden_features[idx], allow_zero_in_degree=True))
        self.edge_weight = args.edge_weight
        self.cls_head = GraphConv(hidden_features[-1], out_features, allow_zero_in_degree=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        # label embedding
        # assume the 0 label is the empty label
        if self.few_shot:
            self.LUT = torch.nn.Embedding(out_features + 1, embedding_dim)
        self.label_weight = args.label_weight
        self.unlabel_weight = args.unlabel_weight

    def label_lookup(self, graph, feats):
        '''
        Look up label embedding
        '''
        node_label = graph.ndata['label']
        label_mask = graph.ndata['label_mask']
        look_up = torch.zeros(feats.shape[0]).to(feats.device)
        for idx, mask in enumerate(label_mask):
            if mask == 1:
                look_up[idx] = node_label[idx] + 1
        label_embedding = self.LUT(look_up.long())
        return label_embedding

    def forward(self, graph, x, edge_weight=None, output_labels=None):
        # check for label embedding
        if self.few_shot == True:
            label_embedding = self.label_lookup(graph, x)
            x = torch.cat([x, label_embedding], dim=1)
        for idx, layer in enumerate(self.backbone):
            if self.edge_weight == True:
                x = layer(graph, x, edge_weight)
            else:
                x = layer(graph, x)
            x = self.relu(x)
        x = self.cls_head(graph, x)
        x = self.softmax(x)
        # loss calculation
        if self.training == True:
            assert (output_labels is not None)
            # class balance loss
            if self.class_balance_loss == True:
                x = self.cls_balance_loss(x, output_labels)
            else:
                x = self.criterion(x, output_labels)
            # weighted loss for few shot
            if self.few_shot:
                x = self.weighted_loss(x, graph)
            else:
                train_mask = graph.ndata["train_mask"]
                x = x[train_mask].mean()
        return x

    def weighted_loss(self, loss, graph):
        '''
        Weighted loss based on label masked
        Node with label given should have more weight
        Node without label given should have less weight
        '''
        label_mask = graph.ndata["label_mask"].reshape(-1)
        train_mask = graph.ndata["train_mask"]
        loss[label_mask == 1] *= self.label_weight
        loss[label_mask == 0] *= self.unlabel_weight
        loss = loss[train_mask].mean()
        return loss

    def cls_balance_loss(self, logits, labels):
        '''
        Weighted loss based on label frequency
        Node with high frequency label should have less weight
        Node with low frequency label should have more weight
        '''
        samples_per_class =  torch.bincount(labels)
        num_classes = logits.shape[1]
        beta = 0.999
        # modified from https://github.com/fcakyon/balanced-loss
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / torch.sum(weights) * num_classes
        weights = torch.tensor(weights, device=logits.device).float()
        loss = F.cross_entropy(logits, labels, weight=weights, reduce=False)
        return loss