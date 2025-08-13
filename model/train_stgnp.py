import os
import copy
import numpy as np
import logging
import torch
import optuna
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from tqdm import tqdm  # Make sure to install tqdm (pip install tqdm)
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_utils import collate
from model.stgnp import STGNP
from utils.train_loop import train_model
from utils.losses import LossODEProcess
from utils.normalisations import MaxMin_Normalization, Ratio_Normalization, SpatialMultivariate_Normalization, compute_norm_info


# Build neural network model
def build_model_multiplex(params, 
                          in_features, 
                          out_dim,
                          num_regions,
                          edge_inter_dim=0,
                          edge_intra_dim=0,
                          external_dim=0,
                          space_planes=5,
                          time_planes=1,
                          depth_nodes=1,
                          depth_edges=1
                          ):
    
    dropout = float(params['dropout'])
    hidden_dim = int(params['hidden_dim'])
    latent_dim = int(params['latent_dim'])
    hidden_dim_ext = int(params['hidden_dim_ext'])
    use_attention = params['use_attention']
    use_hdyn = params['use_hdyn']
    cond_on_time = params['cond_on_time']
    use_constant_edges = params['use_constant_edges']    
    use_regions = params['use_region']
    decode_just_latent = params['decode_just_latent']
    use_time = params['use_time']
    dt_step = params.get('dt_step', 0.1)
    only_spatial = params.get('only_spatial', False)
    use_norm = params.get('use_norm', True)
    use_edges = params.get('use_edges', True)
    use_mse = params.get('use_mse', False)
    classify = params.get('classify', False)
    predict_external = params.get('predict_external', False)
    use_diffusion = params.get('use_diffusion', False)
    use_einsum = params.get('use_einsum', False)
    agg_type = params.get('agg_type', 'sum')
    compute_derivative = params.get('compute_derivative', False)
    use_norm_stmgcn = params.get('use_norm_stmgcn', False)
    use_bias_stmgcn = params.get('use_bias_stmgcn', False)
    space_planes = params.get('space_planes', space_planes)
    time_planes = params.get('time_planes', time_planes)
    depth_nodes = params.get('depth_nodes', depth_nodes)
    depth_edges = params.get('depth_edges', depth_edges)

    model_stm = STGNP(
        in_features, 
        hidden_dim, 
        latent_dim,
        out_dim,
        num_regions,
        space_planes=space_planes,
        time_planes=time_planes,
        edge_inter_dim=edge_inter_dim, 
        edge_intra_dim=edge_intra_dim, 
        external_dim=external_dim,
        hidden_dim_ext=hidden_dim_ext, 
        dropout=dropout,
        use_attention=use_attention,
        use_constant_edges=use_constant_edges,
        use_hdyn=use_hdyn,
        cond_on_time=cond_on_time,
        use_regions=use_regions,
        decode_just_latent=decode_just_latent,
        encode_time=use_time,
        dt_step=dt_step,
        only_spatial=only_spatial,
        use_norm=use_norm,
        use_edges=use_edges,
        use_mse=use_mse,
        classify=classify,
        predict_external=predict_external,
        use_diffusion=use_diffusion,
        use_einsum=use_einsum,
        agg_type=agg_type,
        compute_derivative=compute_derivative,
        use_norm_stmgcn=use_norm_stmgcn,
        use_bias_stmgcn=use_bias_stmgcn,
        depth_nodes=depth_nodes,
        depth_edges=depth_edges,
        )
    
    return model_stm


def encoder_params(model_input):
    param_list = []

    if model_input.hidden_dim_ext > 0:
        for param in model_input.embedding_ext.parameters():
            param_list.append(param)        

    for param in model_input.encoder.parameters():
        param_list.append(param)

    return param_list


def decoder_params(model_input):
    param_list = []

    for param in model_input.decoder.parameters():
        param_list.append(param)

    return param_list


def classifier_params(model_input):
    param_list = []
    for param in model_input.dyn_classifier.parameters():
        param_list.append(param)
    
    # for param in model_input.context_encoder.parameters():
    #     param_list.append(param)

    return param_list


def multiplex_params(model_input):
    param_list = []
    for param in model_input.stgcn.parameters():
        param_list.append(param)
    return param_list


def batch_loop(model: torch.nn.Module,
               data_loader: DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               train_model: bool,
               get_output: bool = False,
               get_latent_state: bool = False,
               **kwargs):
    
    # Get the logger
    logging.getLogger().setLevel(logging.INFO)
        
    epoch = kwargs.get('epoch', 0)
    scaler = kwargs.get('scaler', None)
    predict_beyond = kwargs.get('predict_beyond', False)
    use_position = kwargs.get('use_position', False)
    use_region = kwargs.get('use_region', False)    

    if train_model:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    if get_output:
        list_outputs = []

    if get_latent_state:
        list_latents = []
    
    epoch_metrics = {
        'Total_loss': 0.0,
        'L_rec': 0.0,
        'L_kl': 0.0,
        'L_lat': 0.0,
        'L_graph': 0.0,
        'mse': 0.0,
        'mae': 0.0,
        'mse_pred': 0.0,
        'mae_pred': 0.0,
    }
        
    for ix_batch, batch in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} batches", unit="batch", total=len(data_loader)):
        logging.debug(f'Batch {ix_batch}')

        # Zero the gradients
        if optimizer is not None and train_model:
            optimizer.zero_grad()

        # Get the data        
        graph = batch[0].to(device)
        label = batch[1].to(device)
        
        # Just context time-series data
        in_time, in_node_data, in_node_pos, in_edge_space, in_edge_time, global_data, context_pts, in_pred, region_id, ext_predict_data = batch[2]
        in_time = in_time.to(device).float()
        in_node_data = in_node_data.to(device).float()  # All node node data
        in_node_pos = in_node_pos.to(device).float()
        context_pts = context_pts.to(device) #.float()
        in_pred = in_pred.to(device).float()  # Just thickness and volume

        # All edge data
        in_edge_space = in_edge_space.to(device).float()
        in_edge_time = in_edge_time.to(device).float()

        # Global data and region ID
        if model.hidden_dim_ext > 0:
            global_data = global_data.to(device).float()
            ext_predict_data = ext_predict_data.to(device).float()
        else:
            ext_predict_data = None
            global_data = None
        region_id = region_id.to(device).float()
                                
        # All the time-series data
        tgt_time, tgt_node_data, tgt_node_pos, target_pts, tgt_pred = batch[3]
        tgt_time = tgt_time.to(device).float()
        tgt_node_data = tgt_node_data.to(device).float()  # All node node data
        tgt_node_pos = tgt_node_pos.to(device).float()
        target_pts = target_pts.to(device) #.float()
        tgt_pred = tgt_pred.to(device).float()  # Thickness and volume

        if use_region:
            rids = region_id
        else:
            rids = None
        
        if use_position:
            # Add it to the target data
            tgt_pred = torch.cat([tgt_pred, tgt_node_pos], dim=1)
            pos_data = tgt_node_pos
        else:
            pos_data = None

        #NOTE: For the moment all subjects have the same tgt and context points
        context_pts = context_pts[0]
        target_pts = target_pts[0]

        #NOTE: we use tgt_node_data because then internally we will use the context_pts to get the context data
        output = model(graph, context_pts, target_pts, tgt_time, tgt_node_data, in_edge_space, in_edge_time, rids, pos_data, global_data)
        
        if get_latent_state:
            latent_state = output[-1]
            list_latents.append(latent_state)
            continue

        if get_output:
            # output = output + (tgt_pred, label)
            output = output + (tgt_pred, label, context_pts)
            list_outputs.append(output)
            continue

        # Compute the loss
        # class_data = output[0]  # Classification ouptut
        p_y_pred = output[1]  # Reconstruction output - distribution
        latent_rec = output[2] # Latent space trajectory
        graph_reg = output[3]  # Graph regularization terms
        q_target = output[4]  # Distribution of target latent space
        q_context = output[5]  # Distribution of context latent space
        tgt_edge_distr = output[6]  # Distribution of target edge space
        ctx_edge_distr = output[7]  # Distribution of context edge space

        if kwargs.get('warmup', False):
            T = tgt_pred.shape[-1]
            initial_decay = 0.90  # Starting value (steeper)
            final_decay = 0.999   # Ending value (flatter)
            max_epochs = kwargs.get('warmup_epochs', 100)
            progress = min(epoch / max_epochs, 1.0)
            decay_rate = initial_decay + (final_decay - initial_decay) * progress  # Linear annealing                        
            weights = torch.tensor([decay_rate**t for t in range(T)], device=tgt_pred.device, dtype=torch.float32)
            warmup = True
        else:
            weights = None            
            warmup = False            

        batch_loss, loss_components = criterion(p_y_pred, tgt_pred, 
                                                q_target=q_target, q_context=q_context, 
                                                latent_values=latent_rec, graph_reg=graph_reg, 
                                                tgt_edges=tgt_edge_distr, ctx_edges=ctx_edge_distr,                                                 
                                                warmup=warmup, weights=weights,
                                                rampup_weight=kwargs.get('rampup_weight', 1.0),
                                                )
        
        # Backward and optimize
        if train_model:
            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)  # Unscale before clipping or stepping / only needed for the clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        with torch.no_grad():
            if predict_beyond:
                latent_states = output[-1]                    
                time_to_predict = copy.copy(target_pts)
                time_to_predict = torch.cat([time_to_predict, time_to_predict[[-1]] + 1], dim=0)
                pred_trajectory, pred_latent = model.predict_trajectories(graph, time_to_predict, latent_states, rids=rids, global_data=global_data)

                # Compute MSE and MAE of the predicted vs previous cycle
                mse_pred = (pred_trajectory.mean[..., 1:] - tgt_pred).square().float().sum(dim=-1).mean().item()
                mae_pred = (pred_trajectory.mean[..., 1:] - tgt_pred).abs().float().sum(dim=-1).mean().item()
                epoch_metrics['mse_pred'] += mse_pred
                epoch_metrics['mae_pred'] += mae_pred

                # Output pred
                outout_pred = (pred_trajectory, pred_latent, None)
            else:
                outout_pred = None

            epoch_metrics['Total_loss'] += batch_loss.item()
            for key, val in loss_components.items():
                    epoch_metrics[key] += val #.item()

            # MSE
            mse = (p_y_pred.mean - tgt_pred).square().float().sum(dim=-1).mean().item()

            # MAE
            mae = (p_y_pred.mean - tgt_pred).abs().float().sum(dim=-1).mean().item()

            # Save them            
            epoch_metrics['mse'] += mse
            epoch_metrics['mae'] += mae

    for key in epoch_metrics:
        epoch_metrics[key] /= len(data_loader)

    if get_output:
        return list_outputs

    if get_latent_state:
        return list_latents
    
    # Return the metrics and the last batch output
    return epoch_metrics['Total_loss'], epoch_metrics, output + (tgt_pred,), outout_pred


class Objective_Multiplex(object):
    def __init__(self,
                 study_name,
                 dataset,
                 use_global_data=False,
                 use_edges=True,
                 normalization="NoNorm",
                 norm_by_group=False,
                 norm_only_ed=False,
                 save_dir=None,
                 direction="maximize",
                 device = 'cpu',
                 track_experiment=False,
                 use_position=False,
                 use_region_id=False,
                 use_time=False,
                 use_weighted_sampler=False,
                 space_planes=1,
                 time_planes=1,
                 only_spatial=False,
                 use_norm=True,
                 classify=False,
                 predict_external=False,
                 use_mse=False,
                 use_diffusion=False,
                 use_einsum=False,
                 agg_type='sum',
                 compute_derivative=False,
                 depth_nodes=1,
                 depth_edges=1,
                 fn_batch_loop=batch_loop,
                 ):

        # Hold this implementation specific arguments as the fields of the class.
        self.study_name = study_name
        self.save_dir = save_dir
        self.num_jobs = 1

        self.dataset = dataset
        self.direction = direction
        self.device = device
        self.track_experiment = track_experiment

        # Setting
        self.use_edges = use_edges
        self.use_global_data = use_global_data
        self.normalization = normalization
        self.norm_by_group = norm_by_group
        self.norm_only_ed = norm_only_ed
        self.use_weighted_sampler = use_weighted_sampler
        self.classify = classify
        self.predict_external = predict_external

        # ==== Model setup
        self.use_position = use_position
        self.use_region = use_region_id
        self.use_norm = use_norm
        self.use_time = use_time
        self.space_planes = space_planes
        self.time_planes = time_planes
        self.depth_nodes = depth_nodes
        self.depth_edges = depth_edges
        self.only_spatial = only_spatial
        self.use_mse = use_mse
        self.use_dffusion = use_diffusion
        self.use_einsum = use_einsum
        self.agg_type = agg_type
        self.compute_derivative = compute_derivative
        self.model_setup()
        
        self.batch_loop = fn_batch_loop
        self.direction = direction
        self.track_experiment = track_experiment        
        self.error_score = torch.FloatTensor([-np.inf]).to(device)

    def model_setup(self):
        # Get the dimensions of the problem
        feat_dim = len(self.dataset.list_node_features)
        pos_dim = len(self.dataset.pos_node_features)
        time_dim = len(self.dataset.time_node_features)
        region_dim = len(self.dataset.region_ids)
        
        in_node_dim  = feat_dim
        in_node_dim = in_node_dim + pos_dim if self.use_position else in_node_dim
        in_node_dim = in_node_dim + region_dim if self.use_region else in_node_dim
        in_node_dim = in_node_dim + time_dim if self.use_time else in_node_dim
        self.in_node_dim = in_node_dim
        
        # Out dimensions (trajectory features, not positions)
        self.out_dim = 2  # Median Thickness and Volume Index
        if self.use_position:
            self.out_dim = self.out_dim + pos_dim

        # Number of regions
        num_regions, _, num_frames = self.dataset.nodes_data[0]['nfeatures'].shape    

        # Edge dimensions -- both edges have the same features
        self.edge_intra_dim = len(self.dataset.edges_data[0]['names'])
        self.edge_inter_dim = len(self.dataset.edges_data[0]['names'])
        self.num_regions = num_regions
        self.time_frames = num_frames
        
        # External dimensions, non-graph data    
        self.external_dim = len(self.dataset.list_global_features) if self.use_global_data else 0

    def get_default_params(self):
        default_params = {
            'init_lr': 1e-2,
            'hidden_dim_ext': 3,
            'use_edges': self.use_edges,
            'use_global_data': self.use_global_data,
            'dropout': 0,
            'l1_weight': 0,
            'l2_weight': 0,
            'weight_decay': 0,
            'node_aggr': 'mean',  # 'mean', 'max', 'sum', 'multiple'
            'graph_pooling': 'mlp', # 'mixpool', 'max', 'mean', 'mlp'  # Classifier
            'imbalance_weight': 1,
            'batch_size': 500,
            'num_epochs': 300,
            'loss_balance': 1,
            'use_attention': False,
            'use_constant_edges': False,
            'gamma_rec': 1,
            'gamma_lat': 0,
            'gamma_graph': 0,
            'decode_just_latent': False,
            'space_planes': self.space_planes,
            'time_planes': self.time_planes,
            'depth_nodes': self.depth_nodes,
            'depth_edges': self.depth_edges,
            'use_position': self.use_position,
            'use_region': self.use_region,
            'use_time': self.use_time,
            'use_hdyn': True,
            'cond_on_time': True,
            'weight_classes': None,
            'dt_step': 0.1,
            'only_spatial': self.only_spatial,
            'use_norm': self.use_norm,
            'use_mse': self.use_mse,
            'classify': self.classify,
            'predict_external': self.predict_external,
            'use_diffusion': self.use_dffusion,
            'use_einsum': self.use_einsum,
            'agg_type': self.agg_type,
            'compute_derivative': self.compute_derivative,
            'use_norm_stmgcn': False,
            'use_bias_stmgcn': False,
        }

        return default_params
    
    def set_default_params(self, params={}):
        default_params = self.get_default_params()
        default_params.update(params)
        self.default_params = default_params
    
    def set_indices(self, train_idx, valid_idx, test_idx=None, normal_group_idx=None, save_norm=True):
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        if test_idx is not None:
            self.test_idx = test_idx

        if self.norm_by_group:
            if normal_group_idx is None:
                # By default, use the healthy subjects - ACDC
                healthy_data = self.dataset.global_data.reset_index(drop=True).iloc[train_idx].query("Group=='NOR'").copy()
            else:
                healthy_data = self.dataset.global_data.reset_index(drop=True).iloc[train_idx].query(f"Group=={normal_group_idx}").copy()
            healthy_idx = list(healthy_data.index)
            norm_idx = healthy_idx 
        else:
            norm_idx = self.train_idx

        avg_regions = True  # If True, it will average the regions. If False, it will keep the regions separated

        # Compute the mean and std of the training set // just for the features and positions of the nodes [not time, neiter edge data]
        node_fts = torch.stack([ndata['nfeatures'].clone() for ndata in self.dataset.nodes_data], dim=0)
        norm_info_fts = compute_norm_info(node_fts, norm_idx, only_ed=self.norm_only_ed, avg_regions=avg_regions)
        
        node_pos = torch.stack([ndata['pos'].clone() for ndata in self.dataset.nodes_data], dim=0)
        norm_info_pos = compute_norm_info(node_pos, norm_idx, only_ed=self.norm_only_ed, avg_regions=avg_regions)

        # Compute the mean and std for the edges
        in_edge_space = torch.stack([edata['space'].clone() for edata in self.dataset.edges_data], dim=0)
        norm_info_edge_space = compute_norm_info(in_edge_space, norm_idx, only_ed=self.norm_only_ed, avg_regions=avg_regions)
        
        in_edge_time = torch.stack([edata['time'].clone() for edata in self.dataset.edges_data], dim=0)
        norm_info_edge_time = compute_norm_info(in_edge_time, norm_idx, only_ed=self.norm_only_ed, avg_regions=True)

        # dataset.global_data[['Group', 'Group_Cat']]
        ext_data = self.dataset.global_data.iloc[norm_idx][self.dataset.list_global_features].copy()
        ext_data_clamp = self.dataset.global_data.iloc[self.train_idx][self.dataset.list_global_features].copy()
        norm_info_ext = {
            'mean': torch.tensor(ext_data.mean(axis=0).values).unsqueeze(1).unsqueeze(0),
            'std': torch.tensor(ext_data.std(axis=0).values).unsqueeze(1).unsqueeze(0),
            'min': torch.tensor(np.quantile(ext_data, 0.05, axis=0)).unsqueeze(1).unsqueeze(0),
            'max': torch.tensor(np.quantile(ext_data, 0.95, axis=0)).unsqueeze(1).unsqueeze(0),
            'min_clamp': torch.tensor(np.quantile(ext_data_clamp, 0.05, axis=0)).unsqueeze(1).unsqueeze(0),
            'max_clamp': torch.tensor(np.quantile(ext_data_clamp, 0.95, axis=0)).unsqueeze(1).unsqueeze(0),
        }

        # Set the trasnforms for the dataset
        if self.normalization == "NoNorm":
            data_transforms = {
                'pos': lambda x: x,
                'nfeatures': lambda x: x,
                'global': lambda x: x,
                'space': lambda x: x,
                'time': lambda x: x,
                }
        elif self.normalization == "Spatial":
            data_transforms = {
                'pos': SpatialMultivariate_Normalization(norm_info_pos['mean_data'], norm_info_pos['K1'], norm_info_pos['K2']),
                'nfeatures': SpatialMultivariate_Normalization(norm_info_fts['mean_data'], norm_info_fts['K1'], norm_info_fts['K2']),
                'global': transforms.Normalize(norm_info_ext['mean'], norm_info_ext['std']),
                'space': SpatialMultivariate_Normalization(norm_info_edge_space['mean_data'], norm_info_edge_space['K1'], norm_info_edge_space['K2']),
                'time': SpatialMultivariate_Normalization(norm_info_edge_time['mean_data'], norm_info_edge_time['K1'], norm_info_edge_time['K2']),
                }
        elif self.normalization == "ZNorm":
            data_transforms = {
                'pos': transforms.Normalize(norm_info_pos['mean'], norm_info_pos['std']),
                'nfeatures': transforms.Normalize(norm_info_fts['mean'], norm_info_fts['std']),
                'global': transforms.Normalize(norm_info_ext['mean'], norm_info_ext['std']),
                'space': transforms.Normalize(norm_info_edge_space['mean'], norm_info_edge_space['std']),
                'time': transforms.Normalize(norm_info_edge_time['mean'], norm_info_edge_time['std']),
                }
        elif self.normalization == "MaxMin":
            data_transforms = {
                'pos': MaxMin_Normalization(norm_info_pos['max'], norm_info_pos['min']),
                'nfeatures': MaxMin_Normalization(norm_info_fts['max'], norm_info_fts['min']),
                'global': MaxMin_Normalization(norm_info_ext['max'], norm_info_ext['min']),
                'space': MaxMin_Normalization(norm_info_edge_space['max'], norm_info_edge_space['min']),
                'time': MaxMin_Normalization(norm_info_edge_time['max'], norm_info_edge_time['min']),
                }
        elif self.normalization == "Ratio":
            data_transforms = {
                'pos': Ratio_Normalization(norm_info_pos['mean']),
                'nfeatures': Ratio_Normalization(norm_info_fts['mean']),
                'global': Ratio_Normalization(norm_info_ext['mean']),
                'space': Ratio_Normalization(norm_info_edge_space['mean']),
                'time': Ratio_Normalization(norm_info_edge_time['mean']),
                }
        else:
            raise ValueError(f"Unknown normalization method {self.normalization}")

        # Assign the transforms to the dataset
        self.dataset._transform = data_transforms
        self.norm_info = self.dataset._transform
        if save_norm:
            self.save_norm_info(save_name='norm_info.pt')

    def save_norm_info(self, save_name='norm_info.pt'):
        save_path = os.path.join(self.save_dir, save_name)
        torch.save(self.norm_info, save_path)
    

    def get_latent_state(self, model, dataset, params):
        if params is None:
            params = self.default_params

        with torch.no_grad():
            batch_size = len(dataset)
            dataset_loader = DataLoader(copy.deepcopy(dataset), collate_fn=collate, batch_size=batch_size, shuffle=False)
            model.eval()

            dataset.is_test = False
            latent_states = batch_loop(model, dataset_loader, None, None, self.device, train_model=False, is_test=False, get_latent_state=True, **params)
            return latent_states[0]


    def predict_from_latent(self, model, dataset, time_to_predict, params, device='cpu'):
        if params is None:
            params = self.default_params
        
        use_region = params.get('use_region', False)    

        with torch.no_grad():
            batch_size = len(dataset)
            dataset_loader = DataLoader(copy.deepcopy(dataset), collate_fn=collate, batch_size=batch_size, shuffle=False)
            model.eval()

            dataset.is_test = False
            for ix_batch, batch in enumerate(dataset_loader):
                logging.debug(f'Batch {ix_batch}')

                # Get the dataArchitecture
                graph = batch[0].to(device)
                label = batch[1].to(device)

                # Get the global data and region ID
                # _, _, _, _, _, global_data, _, _, region_id, _ = batch[2]
                in_time, in_node_data, in_node_pos, in_edge_space, in_edge_time, global_data, context_pts, in_pred, region_id, ext_predict_data = batch[2]
                in_time = in_time.to(device)
                in_node_data = in_node_data.to(device)  # All node node data
                in_edge_space = in_edge_space.to(device)
                in_edge_time = in_edge_time.to(device)
                context_pts = context_pts.to(device)
                in_pred = in_pred.to(device)  # Just thickness and volume
                region_id = region_id.to(device)
                global_data = global_data.to(device)


                # All the time-series data
                tgt_time, tgt_node_data, tgt_node_pos, target_pts, tgt_pred = batch[3]
                tgt_time = tgt_time.to(device)
                tgt_node_data = tgt_node_data.to(device)  # All node node data
                tgt_node_pos = tgt_node_pos.to(device)
                target_pts = target_pts.to(device)
                tgt_pred = tgt_pred.to(device)  # Just thickness and volume

                if use_region:
                    rids = region_id
                else:
                    rids = None
                
                if self.use_position:
                    # Add it to the target data
                    tgt_pred = torch.cat([tgt_pred, tgt_node_pos], dim=1)
                    pos_data = tgt_node_pos
                else:
                    pos_data = None

                # Global data and region ID
                if model.hidden_dim_ext > 0:
                    global_data = global_data.to(device)
                else:
                    global_data = None
                region_id = region_id.to(device)

                if use_region:
                    rids = region_id
                else:
                    rids = None
                
                context_pts = context_pts[0]
                target_pts = target_pts[0]

                # latent_states = self.get_latent_state(model, dataset, params)                
                output = model(graph, context_pts, target_pts, tgt_time, tgt_node_data, in_edge_space, in_edge_time, rids, pos_data, global_data)
                latent_states = output[-1]
                
                # time_to_predict = np.append(time_to_predict, time_to_predict[-1] + 1)
                time_to_predict = copy.copy(target_pts)
                time_to_predict = torch.cat([time_to_predict, time_to_predict[[-1]] + 1], dim=0)                
                pred_trajectory, pred_latent = model.predict_trajectories(graph, time_to_predict, latent_states, rids=rids, global_data=global_data)

                mse_pred = (pred_trajectory.mean[..., 1:] - tgt_pred).square().float().mean().item()
                mae_pred = (pred_trajectory.mean[..., 1:] - tgt_pred).abs().float().mean().item()

        return pred_trajectory, pred_latent, None


    def get_output_of_model(self, model, dataset, criterion, optimizer, n_samples=1, params=None):
        if params is None:
            params = self.default_params

        with torch.no_grad():
            batch_size = len(dataset)
            dataset_loader = DataLoader(copy.deepcopy(dataset), collate_fn=collate, batch_size=batch_size, shuffle=False)
            model.eval()
            
            gamma_rec = params.get('gamma_rec', 1)
            gamma_lat = params.get('gamma_lat', 0)
            gamma_graph = params.get('gamma_graph', 0)
            weight_classes = params.get('weight_classes', None)            
            loss_function = LossODEProcess(gamma_rec=gamma_rec, 
                                           gamma_lat=gamma_lat,
                                           gamma_graph=gamma_graph,
                                           weight_classes=weight_classes,
                                           use_mse=False)
            
            # Try getting multiple outputs at different frames
            if n_samples > 1:
                dataset.is_test = False
                list_outputs = []
                for ix in range(n_samples):
                    out_it = batch_loop(model, dataset_loader, loss_function, optimizer, self.device, train_model=False, is_test=False, kwargs=params, get_output=True)
                    list_outputs.append(out_it)

                # Combine the outputs
                #TODO                
                rec_data = torch.stack([out[1] for out in list_outputs], dim=0)            
                latent_rec = torch.stack([out[2] for out in list_outputs], dim=0)

                return (class_data, rec_data, latent_rec)
            else:
                list_outputs = batch_loop(model, dataset_loader, loss_function, optimizer, self.device, train_model=False, is_test=False, get_output=True,
                                          **params)
                
                # Now, put everything in a single
                for ix_out, out in enumerate(list_outputs):
                    class_data_ = out[0]
                    p_y_pred = out[1]
                    latent_rec_ = out[2]
                    graph_reg_ = out[3]                    
                    q_context = out[5]

                    # The edges
                    q_space_ctx, q_time_ctx = out[7]

                    # Predicted global data
                    pred_g_ = out[8]

                    tgt_data_ = out[-3]
                    tgt_label_ = out[-2]
                    context_pts_ = out[-1]

                    rec_ft_ = p_y_pred.mean
                    rec_std_ = p_y_pred.scale

                    if ix_out == 0:
                        class_data = class_data_
                        rec_ft = rec_ft_
                        rec_std = rec_std_
                        latent_rec = latent_rec_
                        graph_reg = graph_reg_
                        tgt_data = tgt_data_
                        tgt_label = tgt_label_
                        context_pts = context_pts_
                        pred_g = pred_g_
                    else:
                        class_data = torch.cat([class_data, class_data_], dim=0)
                        rec_ft = torch.cat([rec_ft, rec_ft_], dim=0)
                        rec_std = torch.cat([rec_std, rec_std_], dim=0)
                        latent_rec = torch.cat([latent_rec, latent_rec_], dim=0)
                        graph_reg = torch.cat([graph_reg, graph_reg_], dim=0)
                        tgt_data = torch.cat([tgt_data, tgt_data_], dim=0)
                        tgt_label = torch.cat([tgt_label, tgt_label_], dim=0)
                        context_pts = torch.cat([context_pts, context_pts_], dim=0)
                        pred_g = torch.cat([pred_g, pred_g_], dim=0)                        
                
                del list_outputs
                del dataset_loader

                return (class_data, rec_ft, rec_std, latent_rec, q_context, graph_reg, q_space_ctx, q_time_ctx, pred_g, tgt_data, tgt_label, context_pts)

    def build_model(self, params):
        """Build the model"""
        return build_model_multiplex(params, 
                                     self.in_node_dim, 
                                     self.out_dim,
                                     self.num_regions,
                                     edge_inter_dim=self.edge_inter_dim,
                                     edge_intra_dim=self.edge_intra_dim,
                                     external_dim=self.external_dim,
                                     space_planes=self.space_planes,
                                     time_planes=self.time_planes,
                                     depth_nodes=self.depth_nodes,
                                     depth_edges=self.depth_edges,
                                     )

    def _train(self, model, params, save_folder, final_model=False, output_probs=False, save_model=True):
        """Train the model"""
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        init_lr = params['init_lr']
        weight_decay = params['weight_decay']

        # Criterion
        gamma_rec = params.get('gamma_rec', 0)
        gamma_lat = params.get('gamma_lat', 0)
        gamma_graph = params.get('gamma_graph', 0)
        weight_classes = params.get('weight_classes', None)        
        loss_function = LossODEProcess(gamma_rec=gamma_rec, 
                                       gamma_lat=gamma_lat,
                                       gamma_graph=gamma_graph,
                                       weight_classes=weight_classes,
                                       use_mse=self.use_mse)
    
        # =================================================================================================
        # OPTIMIZER HERE
        # =================================================================================================        
        # Optimizer
        # Adam / AdamW / RMSprop / Rprop / ASGD
        # optimizer = torch.optim.Adam([{'params': encoder_params(model), 'lr': init_lr},
        #                               {'params': decoder_params(model), 'lr': init_lr},
        #                               {'params': classifier_params(model), 'lr': init_lr},
        #                               {'params': multiplex_params(model), 'lr': init_lr},
        #                               ], lr=init_lr, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay, amsgrad=False, eps=1e-8, betas=(0.9, 0.999))

        # optimizer = torch.optim.RMSprop(model.parameters(), lr=init_lr, alpha=0.99)
        # optimizer = ADPOT(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        # Scheduler
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5, last_epoch=-1)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                                         factor=params.get('lr_reduce_factor', 0.4),
        #                                                         patience=params.get('lr_schedule_patience', 20),
        #                                                         threshold=params.get('lr_schedule_threshold', 1e-4),
        #                                                         min_lr=params.get('lr_min', 1e-6),
        #                                                         )

        # Scheduler with early cut-off factor of 1.15
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int((num_epochs-50) * 1.15), eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)

        # The weighted sampler
        if self.use_weighted_sampler: 
            # dict_groups = self.dataset.global_data[['Group', 'Group_Cat']].drop_duplicates().dropna().set_index(['Group']).to_dict()['Group_Cat']
            # NOR is label 3, the rest are pathological
            labels = self.dataset.label
            num_nor = int((labels == 3).sum())
            num_path = len(labels) - num_nor
            num_total = len(labels)
            weight_path = 1 / (num_path/num_total)
            weight_nor = 1 / (num_nor/num_total)
            class_weights = torch.tensor([weight_path, weight_path, weight_path, weight_nor, weight_path])
            # class_weights = torch.tensor([1, 1, 1, 1, 1])  # No weights

            # The sampler is per dataloader
            train_labels = labels[self.train_idx]
            weights_per_sample = [class_weights[label] for label in train_labels]
            sampler = WeightedRandomSampler(weights=weights_per_sample, num_samples=batch_size, replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        # Datasets
        if final_model:
            reload_model = True            
            train_dataset = Subset(copy.deepcopy(self.dataset), self.train_idx)
            if self.valid_idx is None:
                valid_dataset = None
            else:
                valid_dataset = Subset(copy.deepcopy(self.dataset), self.valid_idx)
            test_dataset = Subset(copy.deepcopy(self.dataset), self.test_idx)
        else:
            reload_model = False
            train_dataset = Subset(copy.deepcopy(self.dataset), self.train_idx)
            valid_dataset = Subset(copy.deepcopy(self.dataset), self.valid_idx)
            test_dataset = None

        # Graph data loader
        # total_cpus = multiprocessing.cpu_count()
        total_cpus = self.num_jobs * 2  # It seems that in SLURM the number of CPUs is not reliable usng mp.cpu_count()
        num_workers_per_dataloader = max(1, (total_cpus // self.num_jobs) // 4)  # Balance CPU usage
        drop_last = False if batch_size >= len(train_dataset) else True
        # drop_last = False
        train_dataloader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=drop_last, sampler=None,
                                      prefetch_factor=2, num_workers=num_workers_per_dataloader, pin_memory=True, persistent_workers=True)
        
        if valid_dataset is not None:
            val_dataloader = DataLoader(valid_dataset, collate_fn=collate, batch_size=len(valid_dataset))
        else:
            val_dataloader = None

        if test_dataset is not None:
            test_dataloader = DataLoader(test_dataset, collate_fn=collate, batch_size=len(test_dataset))    
        else:
            test_dataloader = None        

        # Train the model 
        # Store current handlers before modifying logging inside train_model()
        previous_handlers = logging.getLogger().handlers[:]
        logging.info("Training the model...\n")
        try:
            res_training = train_model(model, 
                                       num_epochs, 
                                       optimizer, 
                                       loss_function, 
                                       train_dataloader, 
                                       test_dataloader, 
                                       val_dataloader=val_dataloader, 
                                       device=self.device, 
                                       save_folder=save_folder, 
                                       reload_model=reload_model, 
                                       scheduler=scheduler, 
                                       early_stop=False, 
                                       early_stop_patience=10, 
                                       tolerance=1e-4, 
                                       print_epoch=10, 
                                       project_name=self.study_name,
                                       track_experiment=self.track_experiment,
                                       l1_weight=params.get('l1_weight', 0.),
                                       l2_weight=params.get('l2_weight', 0.),
                                       output_probs=output_probs, 
                                       batch_loop=self.batch_loop,
                                       hyperparams=params,
                                       gamma_rec=params['gamma_rec'],
                                       gamma_lat=params['gamma_lat'],
                                       gamma_graph=params['gamma_graph'],
                                       use_position=params.get('use_position', False),
                                       use_region=params.get('use_region', False),
                                       error_score=self.error_score,
                                       weight_classes=params.get('weight_classes', None),
                                       save_model=save_model,
                                       warmup_epochs=100,
                                       )
            # Restore previous handlers after train_model() to prevent duplicates
            logger = logging.getLogger()
            logger.handlers = previous_handlers
            logging.info("Model trained!\n")
        except ValueError as e:
            # Restore previous handlers after train_model() to prevent duplicates
            logger = logging.getLogger()
            logger.handlers = previous_handlers
            logging.error(f"Error: {e}")
            print(f"Error: {e}")
            res_training = {'best_score': self.error_score}

        return res_training

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments
        
        params = {
            'hidden_dim': trial.suggest_int("hidden_dim", 12, 20),
            'latent_dim': trial.suggest_int("latent_dim", 2, 10),
            'space_planes': trial.suggest_int("space_planes", 2, 6),
            'time_planes': trial.suggest_int("time_planes", 2, 6),
            'depth_nodes': trial.suggest_int("depth_nodes", 1, 2),
            'depth_edges': trial.suggest_int("depth_edges", 1, 2),
            'weight_decay': trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            'gamma_rec': trial.suggest_float("gamma_rec", 0.5, 1.),
            'gamma_lat': trial.suggest_float("gamma_lat", 0., 0.5),
            'gamma_graph': trial.suggest_float("gamma_graph", 0., 0.5),
            'use_attention': trial.suggest_categorical("use_attention", [True, False]),            
            'decode_just_latent': trial.suggest_categorical("decode_just_latent", [True, False]),
        }

        # Add the default parameters to params if not present already
        config_dict = self.default_params.copy()
        config_dict.update(params)
        
        # Save folder
        save_folder = os.path.join(self.save_dir, f'Trial_{trial.number}')
        os.makedirs(save_folder, exist_ok=True)

        # Model
        model = self.build_model(config_dict)

        # Train it
        try:
            res_training = self._train(model, config_dict, save_folder, final_model=False, save_model=False)
            score = res_training['best_score']
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            score = self.error_score

        # Ensure 'score' is a tensor before calling torch.isclose()
        if isinstance(score, np.ndarray) or isinstance(score, np.float64) or isinstance(score, float):
            score = torch.tensor(score, dtype=torch.float32)

        if torch.isclose(score, self.error_score)[0]:
            # Prevent Optuna from keeping failed trials
            raise optuna.TrialPruned()            
        else:
            # Save the trained model
            score = res_training['best_score']
        
        # Log the best trial after training
        writer = SummaryWriter(log_dir=f"{self.save_dir}/logs")
        writer.add_scalar("Optuna/Score", score, trial.number)
        for key, value in params.items():
            writer.add_scalar(f"Optuna/Params/{key}", value, trial.number)
        writer.close()

        # Free memory
        del model

        return score


