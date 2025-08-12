import os
import logging
import numpy as np
import shutil
import dgl
import torch
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler

from dataset.GraphDataset import GraphDataset


def load_dataset(folder_list, save_folder, reprocess=True, use_global_data=True, mode="Classification",
                 use_prediction=False, is_test=False,  drop_blood_pools=True, use_all=True, use_similarity=False, 
                 noise_lvl=0.0, df_metadata=None):
    # ===================
    # === Note on indices
    # - For nodes
    # Cycle_ID: ID of the cardiac cycle timepoint, equivalent to 'frame'
    # Region_ID: ID of the cardiac region
    # Label_ID: ID of the Cycle+Region
    #
    # - For edges
    # Cycle_1_ID: ID of the cycle instant of the source region
    # Cycle_2_ID: ID of the cycle instant of the target region
    # Target_ID: ID of the target region
    # Source_ID: ID of the source region
    # TargetLabel_ID: ID of the cycle+target region
    # SourceLabel_ID: ID of the cycle+source region
    # ===================
    os.makedirs(save_folder, exist_ok=True)

    # ======= Define dataset name
    conn_str = 'conn-all' if use_all else 'conn-aha'
    sim_str = 'edges-sim' if use_similarity else 'edges-dist'
    bp_str = '' if drop_blood_pools else '_blood_pools'

    dataset_name = f"{conn_str}_{sim_str}{bp_str}"

    # ======= Define data indices
    node_indices = ['Subject', 'Region', 'Label', 'Region_ID', 'Cycle_ID', 'Label_ID']
    edge_indices = ['Subject', 'Source', 'Target', 'Cycle_1', 'Cycle_2', 'SourceLabel', 'TargetLabel',
                    'Source_ID', 'Target_ID', 'Cycle_1_ID', 'Cycle_2_ID', 'TargetLabel_ID', 'SourceLabel_ID',
                    'Edge_Type']
    global_indices = ['Subject', 'Group']
    data_indices = list(set(node_indices + edge_indices + global_indices))        

    if os.path.isdir(os.path.join(save_folder, dataset_name)) and not reprocess:
        # os.path.isdir(os.path.join(save_folder, dataset_name))
        dataset = GraphDataset(name=dataset_name, reprocess=reprocess, save_dir=save_folder, mode=f"{mode}", use_global=use_global_data, 
                               is_test=is_test, use_prediction=use_prediction, edges_conn=conn_str, edges_sim=sim_str, noise_lvl=noise_lvl,
                               metadata=df_metadata, blood_pools=bp_str)
        return dataset    
        
    # Note: if the dataset already exists it will re-load its own train, test, valid indices [if provided previously]
    dataset = GraphDataset(folder_list, data_indices, dataset_name, reprocess, save_folder, mode, use_global_data, 
                           use_prediction, is_test, edges_conn=conn_str, edges_sim=sim_str, noise_lvl=noise_lvl, 
                           metadata=df_metadata, blood_pools=bp_str)

    return dataset


def collate(samples):
    """
    Collate function for graph classification.
    """
    graphs, labels, context_pts, target_pts, input_node, input_edge, target_node, global_data, ext_predict = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)

    # Frame indices
    context_pts = torch.tensor(np.array(context_pts))
    target_pts = torch.tensor(np.array(target_pts))

    # V: vertices, F: features, T: time, B: batch
    # Data will be stacked in format: [B, V, F, T]
    # We want to use the format: [B, F, V, T], s.t. features are channels, for the nodes.

    # Time
    time = torch.stack([ndata['time'] for ndata in input_node], dim=0).permute(0, 2, 1, 3)
    target_time = torch.stack([ndata['time'] for ndata in target_node], dim=0).permute(0, 2, 1, 3)

    # Node data
    in_node_data = torch.stack([ndata['features'] for ndata in input_node], dim=0).permute(0, 2, 1, 3)
    in_node_pos = torch.stack([ndata['pos'] for ndata in input_node], dim=0).permute(0, 2, 1, 3)

    tgt_node_data = torch.stack([ndata['features'] for ndata in target_node], dim=0).permute(0, 2, 1, 3)
    tgt_node_pos = torch.stack([ndata['pos'] for ndata in target_node], dim=0).permute(0, 2, 1, 3)

    # Region ID
    in_region_id = torch.stack([ndata['region_id'] for ndata in input_node], dim=0).permute(0, 2, 1, 3)
    # tgt_region_id = torch.stack([ndata['region_id'] for ndata in target_node], dim=0).permute(0, 2, 1, 3)

    in_predict_data = torch.stack([ndata['predict'] for ndata in input_node], dim=0).permute(0, 2, 1, 3)
    tgt_predict_data = torch.stack([ndata['predict'] for ndata in target_node], dim=0).permute(0, 2, 1, 3)

    # Edge data
    # For the edges, it's [B, E, F, T], where E is the number of edges
    # We want to use the format: [B, F, E, T], s.t. features are channels, for the edges.
    in_edge_space = torch.stack([edata['space'] for edata in input_edge], dim=0).permute(0, 2, 1, 3)
    in_edge_time = torch.stack([edata['time'] for edata in input_edge], dim=0).permute(0, 2, 1, 3)

    # tgt_edge_space = torch.stack([edata['space'] for edata in target_edge], dim=0).permute(0, 2, 1, 3)
    # tgt_edge_time = torch.stack([edata['time'] for edata in target_edge], dim=0).permute(0, 2, 1, 3)

    # Global data
    global_data = torch.vstack(global_data).float()

    # Predict data
    ext_predict_data = torch.vstack(ext_predict).float()

    # Gathter the data
    input_data = (time, in_node_data, in_node_pos, in_edge_space, in_edge_time, global_data, context_pts, in_predict_data, in_region_id[..., 0], ext_predict_data)
    target_data = (target_time, tgt_node_data, tgt_node_pos, target_pts, tgt_predict_data) 

    return batched_graph, labels, input_data, target_data


def collate_xy(samples):
    # Collate the data
    batched_graph, labels, input_data, target_data = collate(samples)

    # Now, make it per subject    
    tgt_time, tgt_node_data, tgt_node_pos, pred_idx, tgt_pred = target_data

    # Just context time-series data
    in_time, in_node_data, in_node_pos, in_edge_space, in_edge_time, global_data, context_pts, in_pred, region_id, ext_predict_data = input_data
                                
    # All the time-series data
    tgt_time, tgt_node_data, tgt_node_pos, target_pts, tgt_pred = target_data

    # Put all the data together, nodes, edges, positions and global data and the label as y
    num_subjects = tgt_node_data.shape[0]

    # X = torch.cat((tgt_node_data, tgt_time, tgt_node_pos), dim=1)
    X = torch.cat((tgt_node_data, tgt_node_pos), dim=1)
    X = X.reshape(num_subjects, -1)

    # Encode the region_ID as different identifiers and add the global dat for each subject
    region_id_enc = torch.tensor(np.arange(0, region_id.shape[1]))
    region_id_enc = region_id_enc.unsqueeze(0).repeat(num_subjects, 1)
    # X = torch.cat((X, global_data, region_id_enc), dim=1)
    X = torch.cat((X, ext_predict_data, region_id_enc), dim=1)

    # Get the edges        
    X_edges = torch.cat([in_edge_space.reshape(num_subjects, -1), 
                         in_edge_time.reshape(num_subjects, -1)], dim=1)

    return X, X_edges, labels


def collate_xy0(samples):
    # Collate the data
    batched_graph, labels, input_data, target_data = collate(samples)

    # Now, make it per subject    
    tgt_time, tgt_node_data, tgt_node_pos, pred_idx, tgt_pred = target_data

    # Just context time-series data
    in_time, in_node_data, in_node_pos, in_edge_space, in_edge_time, global_data, context_pts, in_pred, region_id, ext_predict_data = input_data
                                
    # All the time-series data
    tgt_time, tgt_node_data, tgt_node_pos, target_pts, tgt_pred = target_data

    # Put all the data together, nodes, edges, positions and global data and the label as y
    num_subjects = tgt_node_data.shape[0]

    # X = torch.cat((tgt_node_data, tgt_time, tgt_node_pos), dim=1)
    X = torch.cat((tgt_node_data, tgt_node_pos), dim=1)
    X = X[..., 0].reshape(num_subjects, -1)

    # Encode the region_ID as different identifiers and add the global dat for each subject
    region_id_enc = torch.tensor(np.arange(0, region_id.shape[1]))
    region_id_enc = region_id_enc.unsqueeze(0).repeat(num_subjects, 1)
    # X = torch.cat((X, global_data, region_id_enc), dim=1)
    X = torch.cat((X, ext_predict_data), dim=1)

    # Get the edges        
    X_edges = torch.cat([in_edge_space[..., 0].reshape(num_subjects, -1), 
                         in_edge_time[..., 0].reshape(num_subjects, -1)], dim=1)

    return X, X_edges, labels


def reshape_to_graph(x_input):
    """Reshape the data from [ Batch, Features, Nodes/Edges, Time] to [Batch*Nodes/Edges, Features, Time].
    This is to assign the data to a batched graph strucure in DGL.
    """
    batch_size, ft_dim, nodes_per_frame, time_frames = x_input.shape    
    x_input = x_input.permute(0, 2, 1, 3)  # [batch_size, nodes, features, time_frames]
    x_rshp = x_input.reshape(batch_size*nodes_per_frame, ft_dim, time_frames)
    return x_rshp


def reshape_to_tensor(x_input, batch_size):
    """Reshape the data from [Batch*Nodes/Edges, Features, Time] to [ Batch, Features, Nodes/Edges, Time].
    This is to reformat the data from a batched graph structure in DGL.
    """ 
    batch_times_nodes, ft_dim, time_frames = x_input.shape
    nodes_per_frame = int(batch_times_nodes / batch_size)
    x_rshp = x_input.reshape(batch_size, nodes_per_frame, ft_dim, time_frames)
    x_rshp = x_rshp.permute(0, 2, 1, 3)  # [batch_size, features, nodes, time_frames]
    return x_rshp


def get_data_in_tensor(dataset, idx, device='cpu', just_t0=False):
    collate = collate_xy0 if just_t0 else collate_xy
    dataloader = DataLoader(dataset, batch_size=len(idx), collate_fn=collate)
    # dataloader = DataLoader(dataset, batch_size=len(idx), sampler=SubsetRandomSampler(idx), collate_fn=collate)

    for ix_batch, batch in enumerate(dataloader):
        # Get the data        
        x = batch[0].to(device)
        x_edges = batch[1].to(device)
        label =  batch[2].to(device)

    return x, x_edges, label


def get_data(derivatives_folder:str, 
             wkspc_name:str = 'STMGCN_New', 
             is_test:bool = False,
             reprocess_datasets:bool = False,
             drop_blood_pools:bool = True,
             use_similarity:bool = False,
             use_all:bool = False,
             list_subjects = None,
             df_metadata = None,
             ):
    """Load the dataset."""
    
    use_global_data = True    
    # use_all = False  # Use all edges, or only AHA
    # drop_blood_pools = True  # No blood pools in the graph (although the ones with them are not yet generated...)

    # Get the workspace folder
    workspace_folder = os.path.join(derivatives_folder, f"{wkspc_name}")

    # Load the data and put in a dataset
    graph_data_folder = os.path.join(derivatives_folder, "sa_data_graph")    
    if list_subjects is not None:
        list_folders = [os.path.join(graph_data_folder, x, 'graphs') for x in os.listdir(graph_data_folder) if os.path.isdir(os.path.join(graph_data_folder, x, 'graphs')) and 'sub-' in x and x in list_subjects]
    else:
        list_folders = [os.path.join(graph_data_folder, x, 'graphs') for x in os.listdir(graph_data_folder) if os.path.isdir(os.path.join(graph_data_folder, x, 'graphs')) and 'sub-' in x]    

    #  ==================== Dataset setup ====================    
    type_dset='test' if is_test else 'train'
    
    logging.info(f"Loading the {type_dset} dataset...\n")            
    dataset = load_dataset(list_folders, workspace_folder, reprocess=reprocess_datasets, use_global_data=use_global_data, mode="Classification", 
                           use_prediction=False, is_test=is_test, drop_blood_pools=drop_blood_pools, use_all=use_all, use_similarity=use_similarity, 
                           noise_lvl=0.0, df_metadata=df_metadata)

    logging.info(f"{type_dset} dataset loaded!\n")

    return dataset


def uncompress_graphs(data_folder, list_subjects):
    # Uncompress the graphs.tar.gz file
    for subject in list_subjects:
        subject_folder = os.path.join(data_folder, subject)
        graph_folder = os.path.join(subject_folder, 'graphs')
        shutil.rmtree(graph_folder, ignore_errors=True)
        if not os.path.isdir(graph_folder):
            os.makedirs(graph_folder, exist_ok=True)
            graph_tar = os.path.join(subject_folder, 'graphs.tar.gz')
            if os.path.isfile(graph_tar):
                os.system(f"tar -xzf {graph_tar} -C {graph_folder}")
            else:
                logging.info(f"Graphs for {subject} not found!")


def data_per_subject(dataset: torch.utils.data.Dataset, 
                     use_edges: bool = True,
                     is_test: bool = False,
                    ):
    """Get the data per subject"""
    dataset.nodes_data[0].keys()
    node_fts = torch.stack([ndata['nfeatures'] for ndata in dataset.nodes_data], dim=0)
    node_pos = torch.stack([ndata['pos'] for ndata in dataset.nodes_data], dim=0)
    node_data = torch.cat([node_fts, node_pos], dim=2)
    node_data = node_data.permute(0, 2, 1, 3)

    # Get the edges data
    in_edge_space = torch.stack([edata['space'] for edata in dataset.edges_data], dim=0)
    in_edge_time = torch.stack([edata['time'] for edata in dataset.edges_data], dim=0)
    edge_data = torch.cat([in_edge_space, in_edge_time], dim=1)  # Since both have the same features, we can concatenate them along the edge dimension
    edge_data = edge_data.permute(0, 2, 1, 3)

    # Gloabl data
    global_data = torch.tensor(dataset.global_data[dataset.list_global_features].values)

    # Get the label values
    y = np.nan if is_test else dataset.label.squeeze().data.numpy()

    return node_data, edge_data, global_data, y