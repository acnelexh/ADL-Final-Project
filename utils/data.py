# Data loading and preprocessing
from dgl.dataloading import DataLoader
import dgl
import torch
import numpy as np
import pandas as pd

def read_hemibrain_graph(args):
    def bodyId_to_idx(bodyId):
        return bodyIds.index(bodyId)
    
    traced_neurons = pd.read_csv(args.dataset + '/traced-neurons.csv')
    traced_roi_connections = pd.read_csv(args.dataset + '/traced-roi-connections.csv')
    traced_total_connections = pd.read_csv(args.dataset + '/traced-total-connections.csv')
    num_nodes = 21663
    unique_labels = traced_neurons['instance'].unique()
    
    label_idx_dict = {}
    for i, label in enumerate(unique_labels):
        label_idx_dict[label] = i
    
    labels = [label_idx_dict[label] for label in traced_neurons['instance'].values]
    
    num_classes = traced_neurons['instance'].nunique()

    bodyIds = traced_total_connections['bodyId_post'].unique()
    bodyIds = sorted(bodyIds)
    
    bodyId_idx_dict = {}
    for i, bodyId in enumerate(bodyIds):
        bodyId_idx_dict[bodyId] = i
    
    pre_indexes = np.vectorize(bodyId_idx_dict.get)(traced_total_connections['bodyId_pre'].values)
    post_indexes = np.vectorize(bodyId_idx_dict.get)(traced_total_connections['bodyId_post'].values)
    
    graph = dgl.graph((pre_indexes, post_indexes), num_nodes=num_nodes)

    return graph, num_nodes

def get_dataloader(args):
    graph, num_nodes = read_hemibrain_graph(args)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(num_nodes), sampler,
        batch_size=1024, shuffle=True, drop_last=False, num_workers=4)

    return dataloader