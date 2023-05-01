# Data loading and preprocessing
from dgl.dataloading import DataLoader
import dgl.function as fn
import dgl
import torch
import numpy as np
import pandas as pd
from dgl.nn import EdgeWeightNorm

def read_hemibrain_graph(args):
    # from the notebook
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

    graph.ndata['features'] = torch.stack([graph.in_degrees(), graph.out_degrees()]).T.type(torch.float32)
    graph.ndata['label'] = torch.tensor(labels).type(torch.int64)
    e_weights = torch.tensor(traced_total_connections['weight'].values)
    
    edge_weight = e_weights.type(torch.float32)
    #graph.update_all(fn.u_mul_e('features', 'w', 'm'), fn.sum('m', 'features'))
    
    norm = EdgeWeightNorm(norm='both')
    norm_edge_weight = norm(graph, edge_weight)
    
    graph.edata['w'] = norm_edge_weight
    
    return graph, num_nodes

def get_dataloader(args, hidden=[512, 1024, 512]):
    sample_length = len(hidden) + 1
    graph, num_nodes = read_hemibrain_graph(args)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(sample_length)
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(num_nodes), sampler,
        batch_size=1024, shuffle=True, drop_last=False, num_workers=1)

    return dataloader