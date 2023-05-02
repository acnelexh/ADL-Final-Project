# Data loading and preprocessing
from dgl.dataloading import DataLoader
import dgl.function as fn
import dgl
import torch
import numpy as np
import pandas as pd
from dgl.nn import EdgeWeightNorm
from dgl.data import DGLBuiltinDataset

class HemiBrainGraphDataset(DGLBuiltinDataset):
    def __init__(self, args):
        self.type = type
        # read in data
        traced_neurons = pd.read_csv(args.dataset + '/traced-neurons.csv')
        traced_total_connections = pd.read_csv(args.dataset + '/traced-total-connections.csv')
        #traced_roi_connections = pd.read_csv(args.dataset + '/traced-roi-connections.csv')
        
        self.device = args.device
        self._num_classes = traced_neurons['instance'].nunique()
        
        # build graph
        graph = self._build_graph(traced_total_connections)
        graph = self._add_edge_weight(graph, traced_total_connections)
        graph = self._add_node_feats(graph)
        graph = self._partition_graph(graph)
        
        # preprocess labels
        labels = self._preprocess_labels(traced_neurons, traced_total_connections)
        graph.ndata['label'] = torch.tensor(labels).type(torch.int64)

        # save graph and labels
        self._labels = torch.tensor(labels).type(torch.int64).to(self.device)
        self._g = dgl.reorder_graph(graph).to(self.device)
    
    def _add_node_feats(self, graph):
        '''
        Add node features to graph
        '''
        graph.ndata['feat'] = torch.tensor(self._preprocess_features(graph))
        return graph
    
    def _add_edge_weight(self, graph, traced_total_connections):
        '''
        Assign edge weight to graph
        '''
        # get edge weight
        e_weights = torch.tensor(traced_total_connections['weight'].values)
        edge_weight = e_weights.type(torch.float32) #3413160
        
        # normalize edge weight
        norm = EdgeWeightNorm(norm='both')
        norm_edge_weight = norm(graph, edge_weight)
        
        # assignment
        graph.edata['w'] = norm_edge_weight
        return graph
        
    
    def _partition_graph(self, graph):
        '''
        Partition graph into train/val/test
        '''
        num_nodes = 21663
        train_mask = torch.zeros(num_nodes, dtype=bool)
        val_mask = torch.zeros(num_nodes, dtype=bool)
        test_mask = torch.zeros(num_nodes, dtype=bool)
        
        train_mask[0:15000] = True
        val_mask[15000:16000] = True
        test_mask[16000:] = True
            
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        
        return graph
    
    def _build_graph(self, traced_total_connections):
        '''
        Build graph from traced_total_connections
        '''
        num_nodes = 21663
        bodyIds = traced_total_connections['bodyId_post'].unique()
        bodyIds = sorted(bodyIds)
        
        bodyId_idx_dict = {}
        for i, bodyId in enumerate(bodyIds):
            bodyId_idx_dict[bodyId] = i
        
        pre_indexes = np.vectorize(bodyId_idx_dict.get)(traced_total_connections['bodyId_pre'].values)
        post_indexes = np.vectorize(bodyId_idx_dict.get)(traced_total_connections['bodyId_post'].values)
        graph = dgl.graph((pre_indexes, post_indexes), num_nodes=num_nodes)
        
        return graph
    
    def get_num_classes(self):
        return self._num_classes
    
    def _preprocess_labels(self, traced_neurons, traced_total_connections):
        unique_labels = traced_neurons['instance'].unique()

        label_idx_dict = {}
        for i, label in enumerate(unique_labels):
            label_idx_dict[label] = i
        
        labels = [label_idx_dict[label] for label in traced_neurons['instance'].values]
        return labels
        
    def _preprocess_features(self, graph):
        feats = torch.stack([graph.in_degrees(), graph.out_degrees()]).T.type(torch.float32)
        return feats

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g
    
    def __len__(self):
        return 1
        

def get_hemibrain_split(args):
    dataset = HemiBrainGraphDataset(args)
    graph = dataset[0]
    
    # get split masks
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    # get node features
    feats = graph.ndata['feat']

    # get labels
    labels = graph.ndata['label']
    
    return {"graph": graph,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "feats": feats,
            "labels": labels}