import numpy as np
import pandas as pd
import pickle as pkl
import os

import torch
import torchvision.transforms as transforms

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

# from dataset.dataset_utils import load_dataset


class GraphDataset(DGLDataset):
    def __init__(self,
                 folder_list: list = None,
                 data_indices: list = None,
                 name: str ='GraphDataset',
                 reprocess: bool =False,
                 save_dir:str = os.getcwd(),
                 mode:str = "Classification",
                 use_global: bool = False,
                 use_prediction: bool = False,
                 is_test: bool = False,
                 edges_conn: str = 'conn-aha',  # 'conn-all' or 'conn-aha'
                 edges_sim: str = 'edges-dist',  # 'edges-dist' or 'edges-sim'
                 get_splits: bool = True,
                 noise_lvl: float = 0.0,
                 metadata: pd.DataFrame = None,
                 blood_pools: str = '',  # '_blood_pools' or ''  - If we want to use blood pools
                 ):
        
        # Saving/loading options
        self._name = f"{name}"
        self._save_path = os.path.join(save_dir, self.name)   
        if not os.path.isdir(self._save_path):
            reprocess = True  # For sure, since we don't have the data

        # Training options
        self.is_test = is_test
        self.noise_lvl = noise_lvl
        self.mode = mode   # Classification / Regression        
        self.use_global = use_global
        self.use_prediction = use_prediction   # Use auxiliary prediction to help classification                
        self.data_indices = data_indices
        self.edges_conn = edges_conn
        self.edges_sim = edges_sim
        self.folder_list = folder_list
        self.get_splits = get_splits
        self.metadata = metadata
        self.blood_pools = blood_pools

        # For plotting; the prediction features
        # self.features = ['Thickness', 'Volume', 'Intensity']
        # self.features = ['Thickness', 'Volume', 'Jacobian']
        self.features = ['Thickness', 'Volume']

        if not reprocess:
            self.load()
            self._node_features_id()
            self._edge_features_id()
            self._global_features_id()
            super().__init__(name=self._name, force_reload=reprocess, save_dir=self._save_path)
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
        Select the node features
        """
        self.list_node_features = self.nodes_data[0]['ft_names']
        self.region_ids = self.nodes_data[0]['region_names']
        self.pos_node_features = self.nodes_data[0]['pos_names']
        self.time_node_features = self.nodes_data[0]['time_names']

    def _edge_features_id(self):
        """
        Select the edge features
        """
        self.list_edge_features = self.edges_data[0]['names']

    def _global_features_id(self, list_global_data=[]):
        """
        Select the global features
        """
        to_drop = ['Group_Cat', 'Group_ID', 'Group', 'Cycle', 'ed_frame_idx', 'es_frame_idx', 'es_cycle_time', 'ed_cycle_time', 'dt', 'BMI']
        to_drop2 = ['RV_SV', 'LV_SV', 'RV_Myo_SV', 'LV_Myo_SV', 'RV_Myo_EF', 'LV_Myo_EF', 'LV_Myo_SVI', 'RV_Myo_SVI']
        if len(list_global_data) == 0:
            list_global_data = [x for x in self.global_data.columns if x not in self.data_indices + to_drop + to_drop2]
        self.list_global_features = list_global_data
        # self.list_global_predict = ['LV_EF', 'RV_EF', 'RV_SVI', 'LV_SVI']
        self.list_global_predict = ['LV_EF', 'RV_EF', 'RV_SVI', 'LV_SVI', 'Height', 'Weight', 'BSA']

    def process(self):
        """
        In this dataset I will generate 1 graph per subject. Each node will be a region at a given
        frame (layer). The interactions can be between nodes at the same layer, or nodes at different layers.
        """        
        # ============ Generate the graphs
        graph_collection = []        
        list_ids = []
        global_data = pd.DataFrame()
        node_data = []
        edge_data = []

        for ix, sub_folder in enumerate(self.folder_list):
            # Get the subject id
            # sub = os.path.basename(sub_folder)
            sub = sub_folder.split('/')[-2]

            # Get the data per subject
            graph_filename = os.path.join(sub_folder, f'graph_{self.edges_conn}{self.blood_pools}.bin')

            nodes_filename = os.path.join(sub_folder, f'graph_{self.edges_conn}_{self.edges_sim}{self.blood_pools}.pkl')
            nodes_filename = nodes_filename.replace('edges', 'nodes')

            edges_filename = os.path.join(sub_folder, f'graph_{self.edges_conn}_{self.edges_sim}{self.blood_pools}.pkl')

            if os.path.isfile(graph_filename):
                hg = dgl.load_graphs(graph_filename)[0][0]       
            else:
                print(f"Graph file not found: {graph_filename}")
                continue
            
            # Load the global data
            global_data_filename = os.path.join(os.path.dirname(sub_folder), 'global.parquet')
            if os.path.isfile(global_data_filename):
                global_data_subj = pd.read_parquet(global_data_filename)
            else:
                print(f"Global data file not found: {global_data_filename}")
                continue
                
            if not os.path.isfile(nodes_filename) or not os.path.isfile(edges_filename):
                print(f"Node or edge data file not found: {nodes_filename} or {edges_filename}")
                continue

            # Append everything
            global_data = pd.concat([global_data, global_data_subj], axis=0)
            graph_collection.append(hg)  # Store graph
            list_ids.append(sub)  # Store subject id            

            # Load node and edge data
            # graph_conn-all_nodes-dist.pkl
            with open(nodes_filename, 'rb') as f:
                node_data_subj = pkl.load(f)
            
            #TODO: Verify if any feaure is NaN if, so, replace by 0
            # Be sure it is Thickness related and in a blood pool RV or LV
            node_data_subj['ft_names']
            reg_idx, ft_idx, time_idx = np.where(torch.isnan(node_data_subj['nfeatures']))
            if len(ft_idx) > 0:
                nan_fts = list(np.array(node_data_subj['ft_names'])[np.unique(ft_idx)])
                nan_regions = np.unique(np.array(node_data_subj['region_names'])[np.unique(reg_idx)])
                #TODO: Maybe use more descriptive names, but need to change the CardiacGraph generation process.
                if self.blood_pools == '_blood_pools' and np.all(nan_regions == [['Region_0', 'Region_1']]):
                    # Change NaN to 0 -- 0 thickness / a bit dirty but it is a quick fix
                    node_data_subj['nfeatures'][reg_idx, ft_idx, time_idx] = 0
                else:
                    raise ValueError(f"NaN values found in the node features: {nan_fts} in regions {nan_regions}")
            node_data.append(node_data_subj)  # Store node data 

            # graph_conn-all_edges-dist            
            with open(edges_filename, 'rb') as f:
                edge_data_subj = pkl.load(f)
            edge_data.append(edge_data_subj)  # Store edge data
 
        # Store graphs, labels and subject ids
        # See if 'Group' is in the global data // if not it is a test set
        if not self.is_test:
            try:
                assert 'Group' in global_data.columns, "Group column not found in the test set"                        
            except AssertionError:
                # Check if we have metadata and we can use it to get the group
                if self.metadata is not None:
                    if 'Group' in self.metadata.columns:
                        global_data = global_data.merge(self.metadata[['Subject', 'Group']], on='Subject', how='left')
                        # global_data.groupby(['Group']).count()                        
                    else:
                        raise ValueError("Group column not found in the test set")
                else:
                    raise ValueError("Group column not found in the test set")
            labels = global_data['Group'].astype('category').cat.codes
            global_data['Group_Cat'] = labels
            self.label = torch.from_numpy(np.array(labels).astype(int))
            self.num_classes = len(global_data['Group'].dropna().unique())

        self.graph = graph_collection                    
        self.sub_id = list_ids
        self.global_data = global_data.set_index(['Subject']).copy()
        self.nodes_data = node_data
        self.edges_data = edge_data

        # Get the list of node, edge and global features
        self._node_features_id()
        self._edge_features_id()
        self._global_features_id()

        # Split the data in train, valid and test // used for the fix cross-validation 
        if self.get_splits:
            self.set_train_test_valid_indices()        


    def __getitem__(self, i):
        # ==================
        # Get the data
        # ==================
        try:
            graph = self.graph[i]
            
            if self.is_test:
                label = np.nan
            else:
                label = self.label[i]

            if self.use_global:
                subj_data = self.global_data.loc[self.sub_id[i]].copy()
                global_data = torch.from_numpy(subj_data[self.list_global_features].values.astype(float))
                global_data = self._transform['global'](global_data.unsqueeze(1).unsqueeze(0))
                ext_predict = global_data[0, np.isin(self.list_global_features, self.list_global_predict)].unsqueeze(0)
            else:
                global_data = np.nan
                ext_predict = np.nan

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
            # Get the features to predict
            # ==================
            idx_thickness = np.where(np.isin(self.list_node_features, ['Thickness_Median']))[0]
            idx_volume = np.where(np.isin(self.list_node_features, ['Volume_Index']))[0]
            idx_median = np.where(np.isin(self.list_node_features, ['Intensity_Median']))[0]
            idx_J = np.where(np.isin(self.list_node_features, ['J_Median']))[0]
            fts_to_predict = np.concatenate([idx_thickness, idx_volume])
            # fts_to_predict = np.concatenate([idx_thickness, idx_volume, idx_J])
            # fts_to_predict = np.concatenate([idx_thickness, idx_volume, idx_median])
            # fts_to_predict = np.concatenate([idx_thickness, idx_volume, idx_median, idx_J])

            # ==================
            # Transform the data
            # ==================
            nfeatures = self._transform['nfeatures'](node_data['nfeatures'].permute(2, 0, 1)).permute(1, 2, 0)
            pos_data = self._transform['pos'](node_data['pos'].permute(2, 0, 1)).permute(1, 2, 0)
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
            input_node_data = {'features': nfeatures[...,context_pts],
                            'pos': pos_data[...,context_pts],
                            'time': node_data['time'][...,context_pts],
                            'region_id': node_data['region_id'][...,context_pts],
                            'predict': nfeatures[...,context_pts][:, fts_to_predict],
                            }
            
            # input_edge_data = {'space': edge_space[...,context_pts],
            #                    'time': edge_time[...,context_pts]}
            input_edge_data = {'space': edge_space,
                            'time': edge_time,
                            }
            
            target_node_data = {'features': nfeatures[...,target_pts],
                                'pos': pos_data[...,target_pts],
                                'time': node_data['time'][...,target_pts],
                                'predict': nfeatures[..., target_pts][:, fts_to_predict],
                                }
                    
            return graph, label, context_pts, target_pts, input_node_data, input_edge_data, target_node_data, global_data.squeeze(), ext_predict.squeeze()
        except Exception as e:
            print(f"Error loading index {i}: {e}")
            return None  # Return an empty tensor or raise an exception

    def __len__(self):
        return len(self.graph)

    def set_train_test_valid_indices(self):        
        if self.is_test:
            indices = np.arange(0, len(np.unique(self.global_data.index)))
            splits = None                        
            self.idx_train = None
            self.idx_valid = None
            self.idx_test = None
        else:
            from utils.model_selection_sklearn import stratified_split
            g_subj = self.global_data['Group_Cat'].copy().reset_index().dropna()
            labels = g_subj['Group_Cat'].values
            indices = np.arange(0, len(labels))
            splits = stratified_split(indices, labels, test_size=0.2, valid_size=0.2)

            self.idx_train = splits['X_train']
            self.idx_valid = splits['X_valid']
            self.idx_test = splits['X_test']
            print("=== Number of subjects per split:\n"
                f"\tTrain: {len(splits['X_train'])}\n"
                f"\tTest: {len(splits['X_test'])}\n"
                f"\tValidation: {len(splits['X_valid'])}\n")

            classes, count_per_class_train = np.unique(splits['y_train'], return_counts=True)
            _, count_per_class_valid = np.unique(splits['y_valid'], return_counts=True)
            _, count_per_class_test = np.unique(splits['y_test'], return_counts=True)
            print("=== Counts per class:\n"
                f"Classes: {classes}\n"
                f"Train: {count_per_class_train}\n"
                f"Test: {count_per_class_test}\n"
                f"Valid: {count_per_class_valid}\n")

    def has_cache(self):
        """ check whether there are processed data in 'self._save_path' """
        graph_path = os.path.join(self._save_path, self.name + '_dgl_graph.bin')
        info_path = os.path.join(self._save_path, self.name + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def save(self):
        """ Save the dataset information """
        # Save graphs and labels
        graph_path = os.path.join(self._save_path, self.name + '_dgl_graph.bin')
        if self.is_test:
            save_graphs(graph_path, self.graph,)
        else:
            save_graphs(graph_path, self.graph, {'labels': self.label})

        # Save other information in python dict
        info_path = os.path.join(self._save_path, self.name + '_info.pkl')
        save_info(info_path, {'ids': self.sub_id,
                              'node_data': self.nodes_data,
                              'edge_data': self.edges_data,
                              'glob_data_subj': self.global_data,
                              'idx_train': self.idx_train,
                              'idx_test': self.idx_test,
                              'idx_valid': self.idx_valid,
                              'data_indices': self.data_indices,})

    def load(self):
        """ Load the dataset information """
        # Load graph and labels
        graph_path = os.path.join(self._save_path, self.name + '_dgl_graph.bin')
        self.graph, label_dict = load_graphs(graph_path)
        if not self.is_test:
            self.label = label_dict['labels']

        # Load info
        # conn-aha_edges-dist_info
        info_path = os.path.join(self._save_path, self.name + '_info.pkl')        
        self.sub_id = load_info(info_path)['ids']

        self.nodes_data = load_info(info_path)['node_data']
        self.edges_data = load_info(info_path)['edge_data']
        self.global_data = load_info(info_path)['glob_data_subj']

        self.idx_train = load_info(info_path)['idx_train']
        self.idx_test = load_info(info_path)['idx_test']
        self.idx_valid = load_info(info_path)['idx_valid']
        self.data_indices = load_info(info_path)['data_indices']



if __name__ == '__main__':
    # === Folders
    # Local
    derivatives_folder = '/home/jaume/Desktop/Data/ROI_tmp_train/derivatives'
    # derivatives_folder = '/home/jaume/Desktop/Data/ROI_tmp_test/derivatives'
    workspace_folder = os.path.join(derivatives_folder, "STMGCN")
    os.makedirs(workspace_folder, exist_ok=True)

    # Remote

    # Get the list of folders in the graph_data
    graph_data_folder = os.path.join(derivatives_folder, "sa_data_graph")
    list_folders = [os.path.join(graph_data_folder, x, 'new_graphs') for x in os.listdir(graph_data_folder) if os.path.isdir(os.path.join(graph_data_folder, x, 'new_graphs')) and 'sub-' in x]

    use_all = True  # Use all edges, or only AHA
    drop_blood_pools = True  # No blood pools in the graph (although the ones with them are not yet generated...)
    use_similarity = False  # Use similarity instead of distance

    dataset = load_dataset(list_folders, workspace_folder, reprocess=True, use_global_data=True, mode="Classification", 
                           use_prediction=False, is_test=False, drop_blood_pools=drop_blood_pools, use_all=use_all, use_similarity=use_similarity, noise_lvl=0.0)
    graph = dataset[0]
    print(graph)
    
    # Check the splits
    classes, count_per_class_train = np.unique(dataset.label[dataset.idx_train].squeeze().data.numpy(), return_counts=True)
    _, count_per_class_valid = np.unique(dataset.label[dataset.idx_valid].squeeze().data.numpy(), return_counts=True)
    _, count_per_class_test = np.unique(dataset.label[dataset.idx_test].squeeze().data.numpy(), return_counts=True)
    print("=== Counts per class:\n"
          f"Classes: {classes}\n"
          f"Train: {count_per_class_train}, {count_per_class_train.sum()}\n"
          f"Test: {count_per_class_test}, {count_per_class_test.sum()}\n"
          f"Valid: {count_per_class_valid}, {count_per_class_valid.sum()}\n")

    # all_g = dgl.batch(dataset.graph)
    # print(type(dgl.unbatch(all_g)))
