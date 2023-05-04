# Data loading and preprocessing
from dgl.dataloading import DataLoader
import dgl.function as fn
import dgl
import torch
import numpy as np
import pandas as pd
from dgl.nn import EdgeWeightNorm

class HemiBrainGraphDataset():
    def __init__(self, args):
        # args stuff
        self.random_split = args.random_split
        self.data_split = args.data_split
        self.device = args.device

        # read in nodes=====================================================
        nodes_df = pd.read_csv(args.dataset + '/traced-neurons.csv')
        # remove nodes with empty label
        nodes_df = self._remove_nodes(nodes_df)
        self._build_bodyId_idx_dict(nodes_df)
        self.num_nodes = nodes_df.shape[0]
        self.num_classes = nodes_df['type'].nunique()
        
        # read in labels=====================================================
        labels = self._preprocess_labels(nodes_df)
        self._labels = torch.tensor(labels).type(torch.int64).to(self.device)
        
        # read in edges======================================================
        edge_df = pd.read_csv(args.dataset + '/traced-total-connections.csv')
        # remove edges that are not in the traced neurons df
        edge_df = self._remove_edges(edge_df)
        #traced_roi_connections = pd.read_csv(args.dataset + '/traced-roi-connections.csv')
        
        # build graph
        # TODO make sure graph is correct
        # build graph and add features
        graph = self._build_graph(edge_df)
        graph = self._add_edge_weight(graph, edge_df)
        graph.ndata['label'] = torch.tensor(labels).type(torch.int64)
        graph = self._add_node_feats(graph, args)
        graph = self._partition_graph(graph, random=args.random_split)
        
        self._g = dgl.reorder_graph(graph).to(self.device)
    
    def _remove_nodes(self, df):
        '''
        Remove nodes with empty label
        '''
        df = df.dropna(subset=['type'])
        return df
    
    def _remove_edges(self, df):
        '''
        Remove edges that are not in the traced neurons df
        '''
        existing_nodes = self.bodyId_idx_dict.keys()
        df = df[df['bodyId_pre'].isin(existing_nodes)]
        df = df[df['bodyId_post'].isin(existing_nodes)]
        return df
    
    def _build_bodyId_idx_dict(self, df):
        bodyIds = df['bodyId'].unique()
        bodyIds = sorted(bodyIds)
        
        self.bodyId_idx_dict = {}
        for i, bodyId in enumerate(bodyIds):
            self.bodyId_idx_dict[bodyId] = i
    
    def _add_node_feats(self, graph, args):
        '''
        Add node features to graph
        '''
        # node degree
        degree = torch.tensor(self._preprocess_degree_feats(graph))
        
        # TODO Add more features
        
        # XYZ coordinates features
        synapses = pd.read_csv(args.dataset + '/hemibrain_all_neurons_metrics_polypre_centrifugal_synapses.csv')
        # iterate through the df and add xyz coordinates to graph
        XYZ = torch.zeros(self.num_nodes, 3) 
        exist = 0
        for index, row in synapses.iterrows():
            bodyId = row['bodyid']
            xyz = row[['X', 'Y', 'Z']].values.astype(np.float32)
            # seems like not all the bodyIds are in the graph
            if bodyId in self.bodyId_idx_dict:
                exist += 1
                XYZ[self.bodyId_idx_dict[bodyId], :] = torch.from_numpy(xyz)
        
        # TODO: add ground truth labels to a subset of nodes
        # select a subset of nodes in graph
        # labels = torch.zeros_like
        # give them label
        label = graph.ndata['label'].clone()
        # mask out 90% of the nodes
        mask = torch.rand(self.num_nodes) > 0.1
        label[mask] = -1
        label = label.unsqueeze(1)
        
        print('Number of nodes with xyz coordinates: ', exist)
        print('fraction of nodes with xyz coordinates: ', exist/self.num_nodes)
        # concat degree and xyz coordinates
               
        graph.ndata['feat'] = torch.cat((degree, XYZ, label), dim=1)
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
        
    def _partition_graph(self, graph, random=True):
        '''
        Partition graph into train/val/test
        '''
        def split_data(num_nodes, idx, split_type='train'):
            train_ratio = int(num_nodes * self.data_split[0])
            val_ratio = int(num_nodes * self.data_split[1])
            test_ratio = int(num_nodes * self.data_split[2])
            if split_type == 'train':
                return idx[:train_ratio]
            elif split_type == 'val':
                return idx[train_ratio:train_ratio+val_ratio]
            elif split_type == 'test':
                return idx[-test_ratio:]
        
        train_mask = torch.zeros(self.num_nodes, dtype=bool)
        val_mask = torch.zeros(self.num_nodes, dtype=bool)
        test_mask = torch.zeros(self.num_nodes, dtype=bool)
        
        # split nodes into train/val/test
        if random == True:
            # random assignment
            # pros: more balanced throughout the graph
            # cons: might jeopardize the locality of the graph
            idx = torch.randperm(self.num_nodes)
        else:
            # topological assignment
            # pros: more locality
            # cons: might not be balance
            idx = torch.arange(self.num_nodes)
        train_mask[split_data(self.num_nodes, idx, 'train')] = True
        val_mask[split_data(self.num_nodes, idx, 'val')] = True
        test_mask[split_data(self.num_nodes, idx, 'test')] = True
            
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        
        return graph
    
    def _build_graph(self, edge_df):
        '''
        Build graph from traced_total_connections
        '''

        pre_indexes = np.vectorize(self.bodyId_idx_dict.get)(edge_df['bodyId_pre'].values)
        post_indexes = np.vectorize(self.bodyId_idx_dict.get)(edge_df['bodyId_post'].values)
        
        # TODO Topological sort graph?
        # Maybe use the 3D coordinates for topological sort?
        graph = dgl.graph((pre_indexes, post_indexes), num_nodes=self.num_nodes)
        
        return graph
    
    def get_num_classes(self):
        return self.num_classes
    
    def _preprocess_labels(self, traced_neurons):
        unique_labels = traced_neurons['type'].unique()

        self.label_idx_dict = {}
        for i, label in enumerate(unique_labels):
            self.label_idx_dict[label] = i
        
        labels = [self.label_idx_dict[label] for label in traced_neurons['type'].values]
        return labels
        
    def _preprocess_degree_feats(self, graph):
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

    return graph