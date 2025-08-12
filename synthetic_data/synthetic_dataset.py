import os

import numpy as np
import torch
import dgl
from dgl import load_graphs, save_graphs
from dgl.data.utils import save_info, load_info
from dgl.data import DGLDataset
import torchvision.transforms as transforms



def collate(samples):
    """
    Collate function for graph classification.
    """
    graphs, label, context_pts, target_pts, input_node_data, input_edge_data = map(list, zip(*samples))    

    batched_graph = dgl.batch(graphs)
    # labels = torch.tensor(labels)

    # Frame indices
    context_pts = torch.tensor(np.array(context_pts))
    target_pts = torch.tensor(np.array(target_pts))

    # V: vertices, F: features, T: time, B: batch
    # Data will be stacked in format: [B, V, F, T]
    # We want to use the format: [B, F, V, T], s.t. features are channels, for the nodes.

    # Time
    time = torch.stack([ndata['time'] for ndata in input_node_data], dim=0).permute(0, 2, 1, 3).float()

    # Node data
    in_node_data = torch.stack([ndata['features'] for ndata in input_node_data], dim=0).permute(0, 2, 1, 3).float()

    # Predicted data
    node_predicted = torch.stack([ndata['features_predicted'] for ndata in input_node_data], dim=0).permute(0, 2, 1, 3).float()

    # in_node_pos = torch.stack([ndata['pos'] for ndata in input_node_data], dim=0).permute(0, 2, 1, 3)

    # avg_node_data = in_node_data.mean(dim=0)
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(in_node_data[0, 0, 0, :].detach().numpy())
    # ax.plot(in_node_data[0, 0, 1, :].detach().numpy())
    # # ax.plot(avg_node_data[0, 0, :].detach().numpy())
    # # ax.plot(avg_node_data[0, 1, :].detach().numpy())
    # plt.show()

    # Region ID
    in_region_id = torch.stack([ndata['region_id'] for ndata in input_node_data], dim=0).permute(0, 2, 1).float()

    # Edge data
    # For the edges, it's [B, E, F, T], where E is the number of edges
    # We want to use the format: [B, F, E, T], s.t. features are channels, for the edges.
    in_edge_space = torch.stack([edata['space'] for edata in input_edge_data], dim=0).permute(0, 2, 1, 3).float()
    in_edge_time = torch.stack([edata['time'] for edata in input_edge_data], dim=0).permute(0, 2, 1, 3).float()

    # Global data
    # global_data = torch.vstack(global_data).float()

    # Gathter the data
    # input_data = (time, in_node_data, in_node_pos, in_edge_space, in_edge_time, global_data, in_region_id)
    input_data = (time, in_node_data, in_edge_space, in_edge_time, in_region_id, node_predicted)
    indices = (context_pts, target_pts)

    # return batched_graph, input_data, indices, labels
    return batched_graph, input_data, indices


class SyntheticDataset(DGLDataset):
    """
    Base class for synthetic datasets
    """
    def __init__(self, name, save_dir=None, reprocess=False, has_labels=False, save_info=False):
        # super().__init__(name, url, raw_dir, save_dir, hash_key, force_reload, verbose, transform)
        self._name = f"{name}"
        self.has_labels = has_labels
        self.save_info = save_info
        self._save_path = os.path.join(save_dir, self.name)   
        if not os.path.isdir(self._save_path):
            reprocess = True  # For sure, since we don't have the data
        
        if not reprocess:
            self.load()
            super().__init__(name=self._name, force_reload=reprocess, save_dir=self._save_path)
            self._node_features_id()
            self._edge_features_id()
            self._global_features_id()
            self._transform = {
                'pos': transforms.Normalize(mean=0, std=1), 
                'nfeatures': transforms.Normalize(mean=0, std=1),
                'global': transforms.Normalize(mean=0, std=1),
                'space': transforms.Normalize(mean=0, std=1),
                'time': transforms.Normalize(mean=0, std=1),
                }
            return
        
        # Initialize
        super().__init__(name=self._name, force_reload=reprocess, save_dir=self._save_path)
        # Automatically triggers: has_cache() -> process(), save() if cache not found
        # If found -> load() and not self._force_reload 
        # It also depends on the status of the self._force_reload flag
        self._transform = {
            'pos': transforms.Normalize(mean=0, std=1), 
            'nfeatures': transforms.Normalize(mean=0, std=1),
            'global': transforms.Normalize(mean=0, std=1),
            'space': transforms.Normalize(mean=0, std=1),
            'time': transforms.Normalize(mean=0, std=1),
            }

    def _node_features_id(self):
        """
        Set the node features names as a list
        """
        num_features = self.graph[0].ndata['nfeatures'].shape[1]
        num_regions = self.graph[0].ndata['region_id'].shape[1]
        num_time_dim = self.graph[0].ndata['time'].shape[1]

        self.list_node_features = [f'feature_{i}' for i in range(num_features)]
        self.time_node_features = [f'time_{i}' for i in range(num_time_dim)]
        self.region_ids = [f'region_{i}' for i in range(num_regions)]

    def _edge_features_id(self):
        """
        Set the edge features names as a list
        """
        num_edge_features = self.graph[0]['space'].edata['cat'].shape[1]
        self.list_edge_features = [f'edge_feature_{i}' for i in range(num_edge_features)]

    def _global_features_id(self):
        """
        Set the global features names as a list
        """
        pass

    def process(self):
        """
        Process the dataset, important to define the self.graph list as a list of graphs
        """
        # Get the list of node, edge and global features
        self._node_features_id()
        self._edge_features_id()
        self._global_features_id()
        pass
        
    def __getitem__(self, idx):
        """
        Return the graph at index idx
        """
        graph = self.graph[idx]

        # Get the node and edge data
        node_data = self.nodes_data[i]
        edge_data = self.edges_data[i]

        # ==================
        # Get the context points
        # ==================
        num_context = 20   # These are like 'control' points        
        total_points = node_data['nfeatures'].shape[-1]
        # Get the set of context points
        # num_subjects = in_data.shape[0]
        # context_pts = np.zeros((num_subjects, num_context+1), dtype=int)
        # for s in range(0, num_subjects):
        #     s_context_pts = np.random.choice(np.arange(1, total_points-1), num_context, replace=False)
        #     s_context_pts = np.concatenate([np.array([0]), s_context_pts])  # Add always intial point to the context
        #     context_pts[s, :] = s_context_pts

        # context_pts = np.random.choice(np.arange(1, total_points-1), num_context, replace=False)
        # context_pts = np.concatenate([np.array([0]), context_pts])  # Add always intial point to the context
        context_pts = np.arange(0, total_points+1, 3)  # Fix context points
        target_pts = np.arange(0, total_points)  # All of them

        # ==================
        # Transform the data
        # ==================
        nfeatures = self._transform['nfeatures'](node_data['nfeatures'].permute(2, 0, 1)).permute(1, 2, 0)
        # pos_data = self._transform['pos'](node_data['pos'].permute(2, 0, 1)).permute(1, 2, 0)

        # pos_data = node_data['pos']
        # pos_data = pos_data[:, :2]  # Don't use the Z-coordinate
        # import matplotlib.pyplot as plt
        # import matplotlib; matplotlib.use('TkAgg')
        # x = node_data['pos'][10][0]
        # y = node_data['pos'][10][1]
        # z = node_data['pos'][10][2]
        # fig, ax = plt.subplots(1, 3)
        # ax[0].plot(x, y); ax[1].plot(x, z); ax[2].plot(y, z); plt.show()

        # Transform the edge data -- Don't transform the edges
        edge_space = edge_data['space']
        edge_time = edge_data['time']
        # edge_space = self._transform['space'](edge_data['space'].permute(2, 0, 1)).permute(1, 2, 0)
        # edge_time = self._transform['time'](edge_data['time'].permute(2, 0, 1)).permute(1, 2, 0)

        # Input data
        input_node_data = {'features': nfeatures,
                        #    'pos': pos_data,
                           'time': node_data['time'],
                           'region_id': node_data['region_id'],
                        #    'predict': nfeatures[:, fts_to_predict],
                           }
                
        input_edge_data = {'space': edge_space,
                           'time': edge_time,
                           }

        if self.has_labels:
            label = self.label[idx]
        else:
             label = np.nan
        
        return graph, label, context_pts, target_pts, input_node_data, input_edge_data

    def __len__(self):
        return len(self.graph)
    
    def has_cache(self):
        """ check whether there are processed data in 'self._save_path' """
        graph_path = os.path.join(self._save_path, self.name + '_dgl_graph.bin')
        return os.path.exists(graph_path)
    
    def save(self):
        """ Save the dataset information """
        # Save graphs and labels
        graph_path = os.path.join(self._save_path, self.name + '_dgl_graph.bin')
        if self.has_labels:
            save_graphs(graph_path, self.graph, {'labels': self.label})            
        else:
            save_graphs(graph_path, self.graph,)

        # Save other information in python dict [optionally]
        info_path = os.path.join(self._save_path, self.name + '_info.pkl')
        if self.save_info:
            save_info(info_path, {'ids': self.sub_id,
                                'node_data': self.nodes_data,
                                'edge_data': self.edges_data,
                                'glob_data_subj': self.global_data,})


    def load(self):
        """ Load the dataset information """
        # Load graph and labels
        graph_path = os.path.join(self._save_path, self.name + '_dgl_graph.bin')        
        if self.has_labels:
            self.graph, label_dict = load_graphs(graph_path)
            self.label = label_dict['labels']
        else:
            self.graph = load_graphs(graph_path)[0]

        # Load info
        info_path = os.path.join(self._save_path, self.name + '_info.pkl')
        if os.path.isfile(info_path):
            self.sub_id = load_info(info_path)['ids']
            self.nodes_data = load_info(info_path)['node_data']
            self.edges_data = load_info(info_path)['edge_data']
            self.global_data = load_info(info_path)['glob_data_subj']
