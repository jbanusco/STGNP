import logging
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # Make sure to install tqdm (pip install tqdm)
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from synthetic_data.synthetic_dataset import collate
from utils.losses import FocalLoss, LossODEProcess
from model.train_stmgcn_ode import Objective_Multiplex
from utils.normalisations import compute_norm_info, MaxMin_Normalization, Ratio_Normalization, SpatialMultivariate_Normalization
from utils.train_loop import train_model
from dataset.dataset_utils import reshape_to_graph, reshape_to_tensor


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

    #TODO - Important!!: Need to add these stuff!!
    # Check if external data is used or not    
    apply_penalties = kwargs.get('apply_penalties', True)    
    epoch = kwargs.get('epoch', 0)
    scaler = kwargs.get('scaler', None)
    predict_beyond = kwargs.get('predict_beyond', False)
    # add_graph = kwargs.get('add_graph', False)
    # writer = kwargs.get('writer', None)

    # Penalty weights
    l1_weight = kwargs.get('l1_weight', 0.0)
    l2_weight = kwargs.get('l2_weight', 0.0)
    use_position = kwargs.get('use_position', False)
    use_region = kwargs.get('use_region', False)    

    # Output additional information in the test loop
    output_probs = kwargs.get('output_probs', False)

    # Re-start scores
    num_params = sum([len(p) for p in model.parameters()])

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
        'L_class': 0.0,
        'L_kl': 0.0,
        'L_bc': 0.0,
        'L_lat': 0.0,
        'L_graph': 0.0,
        'accuracy': 0.0,
        'mse': 0.0,
        'mae': 0.0,
        'mse_pred': 0.0,
        'mae_pred': 0.0,
    }

    # Wrap forward pass and loss computation in autocast for mixed precision.
    # with torch.amp.autocast(device):
    
    # Loop over the batches
    for ix_batch, batch in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} batches", unit="batch", total=len(data_loader)):
        logging.debug(f'Batch {ix_batch}')

        # Zero the gradients
        if optimizer is not None and train_model:
            optimizer.zero_grad()
            
        # Get the data        
        graph = batch[0].to(device)
        
        # Just context time-series data            
        in_time, in_node_data, in_edge_space, in_edge_time, region_id, node_predicted = batch[1]
        in_time = in_time.to(device).float()
        in_node_data = in_node_data.to(device).float()  # All node node data
        # in_node_pos = in_node_pos.to(device)
        node_predicted = node_predicted.to(device).float()
        in_node_pos = None

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # in_node_data_cpu = in_node_data.cpu().detach().numpy()

        # ix_sample = 0
        # ix_node = 0
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(in_node_data_cpu[ix_sample, ix_node, 0, :])
        # ax.plot(in_node_data_cpu[ix_sample, ix_node, 1, :])
        # plt.show()

        # All edge data
        in_edge_space = in_edge_space.to(device).float()
        in_edge_time = in_edge_time.to(device).float()

        # Global data and region ID
        # global_data = global_data.to(device)
        region_id = region_id.to(device) #.float()

        # Indices
        context_pts, target_pts = batch[2]
        context_pts = context_pts.to(device) #.float()
        target_pts = target_pts.to(device) #.float()

        if use_region:
            rids = region_id
        else:
            rids = None
        
        if use_position:
            # Add it to the target data
            tgt_pred = torch.cat([in_node_data, in_node_pos], dim=1).float()
            pos_data = in_node_pos.float()
        else:
            tgt_pred = in_node_data
            pos_data = None

        #NOTE: For the moment all subjects have the same tgt and context points
        context_pts = context_pts[0]
        target_pts = target_pts[0]

        #NOTE: we use tgt_node_data because then internally we will use the context_pts to get the context data
        global_data = None
        output = model(graph, context_pts, target_pts, in_time, in_node_data, in_edge_space, in_edge_time, rids, pos_data, global_data)
        # if not train_model and add_graph:
        #     input_model = (graph, context_pts, target_pts, in_time, in_node_data, in_edge_space, in_edge_time, rids, pos_data, global_data)
        #     writer.add_graph(model, input_model)

        if get_latent_state:
            latent_state = output[-1]
            list_latents.append(latent_state)
            continue

        if get_output:
            # output = output + (tgt_pred, label)
            output = output + (tgt_pred, None, context_pts)
            list_outputs.append(output)
            continue

        # Compute the loss
        class_data = output[0]  # Classification ouptut
        p_y_pred = output[1]  # Reconstruction output - distribution
        latent_rec = output[2] # Latent space trajectory
        graph_reg = output[3]  # Graph regularization terms
        q_target = output[4]  # Distribution of target latent space
        q_context = output[5]  # Distribution of context latent space
        tgt_edge_distr = output[6]  # Distribution of target edge space
        ctx_edge_distr = output[7]  # Distribution of context edge space
        prototypes = model.prototypes

        label = None
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
            # weights = torch.tensor([1.0 for t in range(tgt_pred.shape[-1])], device=tgt_pred.device)
            warmup = False
            # print("No warmup")

        batch_loss, loss_components = criterion(p_y_pred, tgt_pred, class_data, label, q_target=q_target, q_context=q_context, 
                                                latent_values=latent_rec, graph_reg=graph_reg, 
                                                tgt_edges=tgt_edge_distr, ctx_edges=ctx_edge_distr, 
                                                only_rec=kwargs.get('training_regression', False), 
                                                prototypes=prototypes, warmup=warmup,
                                                weights=weights,
                                                rampup_weight=kwargs.get('rampup_weight', 1.0),
                                                )

        # Penalties
        # l1_penalty = l1_weight * sum([p.abs().sum() for p in model.parameters()]) / num_params
        # l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()]) / num_params
        # if apply_penalties:
        #     batch_loss += l1_penalty + l2_penalty

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
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
                optimizer.step()

        with torch.no_grad():
            if predict_beyond:
                latent_states = output[-1]                    
                time_to_predict = copy.copy(target_pts)
                time_to_predict = torch.cat([time_to_predict, time_to_predict[[-1]] + 1], dim=0)
                pred_trajectory, pred_latent = model.predict_trajectories(graph, time_to_predict, latent_states, rids=rids, global_data=global_data)

                # Compute MSE and MAE of the predicted vs node_predicted
                mse_pred = (pred_trajectory.mean[..., 1:] - node_predicted).square().float().sum(dim=-1).mean().item()
                mae_pred = (pred_trajectory.mean[..., 1:] - node_predicted).abs().float().sum(dim=-1).mean().item()
                epoch_metrics['mse_pred'] += mse_pred
                epoch_metrics['mae_pred'] += mae_pred

                # Output pred
                outout_pred = (pred_trajectory, pred_latent, node_predicted)
            else:
                outout_pred = None

            epoch_metrics['Total_loss'] += batch_loss.float().item()
            for key, val in loss_components.items():
                    epoch_metrics[key] += val #.item()

            # Accuracy
            predicted = None
            accuracy = 0.
            # _, predicted = torch.max(class_data.data, 1)
            # correct = (predicted == label.squeeze()).sum().item()
            # accuracy = correct / len(label.squeeze())
            # total_acc += accuracy 

            # MSE
            mse = (p_y_pred.mean - tgt_pred).square().float().sum(dim=-1).mean().item()

            # MAE
            mae = (p_y_pred.mean - tgt_pred).abs().float().sum(dim=-1).mean().item()

            epoch_metrics['accuracy'] += accuracy
            epoch_metrics['mse'] += mse
            epoch_metrics['mae'] += mae

    for key in epoch_metrics:
        epoch_metrics[key] /= len(data_loader)

    if get_output:
        return list_outputs

    if output_probs:
        decisions = [predicted, label.squeeze()]
        probs = class_data
        return epoch_metrics['Total_loss'], epoch_metrics, decisions, probs
    
    if get_latent_state:
        return list_latents

    # Return the metrics and the last batch output
    return epoch_metrics['Total_loss'], epoch_metrics, output + (tgt_pred,), outout_pred



class ObjectiveSynthetic(Objective_Multiplex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_setup(self):
        # Get the dimensions of the problem
        feat_dim = len(self.dataset.list_node_features)
        if self.use_position:
            pos_dim = len(self.dataset.pos_node_features)
        time_dim = len(self.dataset.time_node_features)
        region_dim = len(self.dataset.region_ids)
        
        in_node_dim  = feat_dim
        in_node_dim = in_node_dim + pos_dim if self.use_position else in_node_dim
        in_node_dim = in_node_dim + region_dim if self.use_region else in_node_dim
        in_node_dim = in_node_dim + time_dim if self.use_time else in_node_dim    
        self.in_node_dim = in_node_dim
                
        self.out_dim = feat_dim  # Autoencoder-like

        # Number of regions
        num_regions, _, num_frames = self.dataset.graph[0].ndata['nfeatures'].shape
        self.num_regions = num_regions
        self.time_frames = num_frames        

        # Edge dimensions -- both edges have the same features
        self.edge_intra_dim = len(self.dataset.list_edge_features)
        self.edge_inter_dim = len(self.dataset.list_edge_features)

        # No external data
        self.external_dim = 0

        # Use MSE not variational
        # self.use_mse = True
        # self.use_norm = True

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

                # Get the data
                graph = batch[0].to(device)
                # label = batch[1].to(device)

                # Get the global data and region ID
                in_time, in_node_data, in_edge_space, in_edge_time, region_id, node_predicted = batch[1]
                in_time = in_time.to(device)
                in_node_data = in_node_data.to(device)  # All node node data
                in_edge_space = in_edge_space.to(device)
                in_edge_time = in_edge_time.to(device)
                region_id = region_id.to(device)
                node_predicted = node_predicted.to(device)

                # Indices
                context_pts, target_pts = batch[2]
                context_pts = context_pts.to(device)
                target_pts = target_pts.to(device)

                #NOTE: For the moment all subjects have the same tgt and context points
                context_pts = context_pts[0]
                target_pts = target_pts[0]

                pos_data = None
                global_data = None
                region_id = region_id.to(device)

                if use_region:
                    rids = region_id
                else:
                    rids = None
                
                # latent_states = self.get_latent_state(model, dataset, params)
                # latent_states = batch_loop(model, dataset_loader, None, None, self.device, train_model=False, is_test=False, get_latent_state=True, **params)
                output = model(graph, context_pts, target_pts, in_time, in_node_data, in_edge_space, in_edge_time, rids, pos_data, global_data)
                latent_states = output[-1]
                # latent_rec_ = output[2]
                # a = reshape_to_tensor(latent_states['h'].unsqueeze(-1), batch_size=batch_size)

                # time_to_predict = np.append(time_to_predict, time_to_predict[-1] + 1)
                time_to_predict = torch.cat([time_to_predict, time_to_predict[[-1]] + 1], dim=0)
                pred_trajectory, pred_latent = model.predict_trajectories(graph, time_to_predict, latent_states, rids=rids, global_data=global_data)
                # ((pred_latent[..., 0] - latent_rec_[..., -2])**2).sum()

                # print(((pred_latent[..., 0] - latent_rec_[..., -1])**2).sum())                
                # print(((a[..., 0] - latent_rec_[..., -1])**2).sum())
                # (((pred_latent[..., 0] - a[..., 0])**2).sum())
                # pred_trajectory.scale = pred_trajectory.scale[..., 1:]
                # pred_trajectory.mean = pred_trajectory.mean[..., 1:]
                # pred_latent = pred_latent[..., 1:]

        return pred_trajectory, pred_latent, node_predicted
    
    def get_output_of_model(self, model, dataset, criterion, optimizer, n_samples=1, params=None):
        if params is None:
            params = self.default_params

        with torch.no_grad():
            # dataset.is_test = True
            # batch_size = len(dataset) if len(dataset) < 200 else 200
            batch_size = len(dataset)
            dataset_loader = DataLoader(copy.deepcopy(dataset), collate_fn=collate, batch_size=batch_size, shuffle=False)
            model.eval()
            
            gamma_rec = params.get('gamma_rec', 0)
            gamma_lat = params.get('gamma_lat', 0)
            gamma_class = params.get('gamma_class', 0)
            gamma_bc = params.get('gamma_bc', 0)
            gamma_graph = params.get('gamma_graph', 0)
            weight_classes = params.get('weight_classes', None)
            gamma_focal = params.get('gamma_focal', 0)
            loss_function = LossODEProcess(gamma_rec=gamma_rec,
                                           gamma_lat=gamma_lat,
                                           gamma_class=gamma_class,
                                           gamma_bc=gamma_bc,
                                           gamma_graph=gamma_graph,
                                           weight_classes=weight_classes, gamma_focal=gamma_focal)
            
            # Try getting multiple outputs at different frames
            if n_samples > 1:
                dataset.is_test = False
                list_outputs = []
                for ix in range(n_samples):
                    out_it = batch_loop(model, dataset_loader, loss_function, optimizer, self.device, train_model=False, is_test=False, kwargs=params, get_output=True)
                    list_outputs.append(out_it)

                # Combine the outputs
                #TODO
                class_data = torch.stack([out[0] for out in list_outputs], dim=0)
                rec_data = torch.stack([out[1] for out in list_outputs], dim=0)            
                latent_rec = torch.stack([out[2] for out in list_outputs], dim=0)

                # class_data = output[0]  # Classification ouptut
                # p_y_pred = output[1]  # Reconstruction output - distribution
                # latent_rec = output[2] # Latent space trajectorys
                # force = output[3]  # Force of the system
                # q_target = output[4]  # Distribution of target latent space
                # q_context = output[5]  # Distribution of context latent space    
                # tgt_data = output[-1]

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
                    # q_target = out[4]
                    q_context = out[5]
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
                    else:
                        class_data = torch.cat([class_data, class_data_], dim=0)
                        rec_ft = torch.cat([rec_ft, rec_ft_], dim=0)
                        rec_std = torch.cat([rec_std, rec_std_], dim=0)
                        latent_rec = torch.cat([latent_rec, latent_rec_], dim=0)
                        graph_reg = torch.cat([graph_reg, graph_reg_], dim=0)
                        tgt_data = torch.cat([tgt_data, tgt_data_], dim=0)
                        tgt_label = torch.cat([tgt_label, tgt_label_], dim=0)
                        context_pts = torch.cat([context_pts, context_pts_], dim=0)
                
                del list_outputs
                del dataset_loader

                return (class_data, rec_ft, rec_std, latent_rec, q_context, graph_reg, tgt_data, tgt_label, context_pts)


    def set_indices(self, train_idx, valid_idx, test_idx=None, save_norm=True):
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        if test_idx is not None:
            self.test_idx = test_idx
        norm_idx = self.train_idx

        # Norm data expects a dataframe: [B, V, F, T]
        avg_regions = True  # If True, it will average the regions. If False, it will keep the regions separated
        # Compute the mean and std of the training set // just for the features and positions of the nodes [not time, neiter edge data]
        node_fts = torch.stack([ndata['nfeatures'] for ndata in self.dataset.nodes_data], dim=0)
        norm_info_fts = compute_norm_info(node_fts, norm_idx, only_ed=self.norm_only_ed, avg_regions=avg_regions)
        
        if self.use_position:
            node_pos = torch.stack([ndata['pos'] for ndata in self.dataset.nodes_data], dim=0)
            norm_info_pos = compute_norm_info(node_pos, norm_idx, only_ed=self.norm_only_ed)

        # Compute the mean and std for the edges
        in_edge_space = torch.stack([edata['space'] for edata in self.dataset.edges_data], dim=0)
        norm_info_edge_space = compute_norm_info(in_edge_space, norm_idx, only_ed=self.norm_only_ed, avg_regions=avg_regions)
        
        in_edge_time = torch.stack([edata['time'] for edata in self.dataset.edges_data], dim=0)
        norm_info_edge_time = compute_norm_info(in_edge_time, norm_idx, only_ed=self.norm_only_ed, avg_regions=avg_regions)

        # External information
        if self.use_global_data:
            ext_data = self.dataset.global_data.iloc[norm_idx][self.dataset.list_global_features]
            ext_data_clamp = self.dataset.global_data.iloc[self.train_idx][self.dataset.list_global_features]
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
                'pos': transforms.Normalize(0, 1),
                'nfeatures': transforms.Normalize(0, 1),
                'global': transforms.Normalize(0, 1),
                'space': transforms.Normalize(0, 1),
                'time': transforms.Normalize(0, 1),
                }
        elif self.normalization == "Spatial":
            data_transforms = {
                # 'pos': SpatialMultivariate_Normalization(norm_info_pos['mean_data'], norm_info_pos['K1'], norm_info_pos['K2']),
                'nfeatures': SpatialMultivariate_Normalization(norm_info_fts['mean_data'], norm_info_fts['K1'], norm_info_fts['K2']),
                # 'global': transforms.Normalize(norm_info_ext['mean'], norm_info_ext['std']),
                'space': SpatialMultivariate_Normalization(norm_info_edge_space['mean_data'], norm_info_edge_space['K1'], norm_info_edge_space['K2']),
                'time': SpatialMultivariate_Normalization(norm_info_edge_time['mean_data'], norm_info_edge_time['K1'], norm_info_edge_time['K2']),
                }
        elif self.normalization == "ZNorm":
            data_transforms = {
                # 'pos': transforms.Normalize(norm_info_pos['mean'], norm_info_pos['std']),
                'nfeatures': transforms.Normalize(norm_info_fts['mean'], norm_info_fts['std']),
                # 'global': transforms.Normalize(norm_info_ext['mean'], norm_info_ext['std']),
                'space': transforms.Normalize(norm_info_edge_space['mean'], norm_info_edge_space['std']),
                'time': transforms.Normalize(norm_info_edge_time['mean'], norm_info_edge_time['std']),
                }
        elif self.normalization == "MaxMin":
            data_transforms = {
                # 'pos': MaxMin_Normalization(norm_info_pos['max'], norm_info_pos['min']),
                'nfeatures': MaxMin_Normalization(norm_info_fts['max'], norm_info_fts['min']),
                # 'global': MaxMin_Normalization(norm_info_ext['max'], norm_info_ext['min']),
                'space': MaxMin_Normalization(norm_info_edge_space['max'], norm_info_edge_space['min']),
                'time': MaxMin_Normalization(norm_info_edge_time['max'], norm_info_edge_time['min']),
                }
        elif self.normalization == "Ratio":
            data_transforms = {
                # 'pos': Ratio_Normalization(norm_info_pos['mean']),
                'nfeatures': Ratio_Normalization(norm_info_fts['mean']),
                # 'global': Ratio_Normalization(norm_info_ext['mean']),
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

    def _train(self, model, params, save_folder, final_model=False, output_probs=False, save_model=True, model_to_load=None, fine_tune_model=False):
        """Train the model"""
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        init_lr = params['init_lr']
        weight_decay = params['weight_decay']

        # Criterion
        gamma_rec = params.get('gamma_rec', 0)
        gamma_lat = params.get('gamma_lat', 0)
        gamma_class = params.get('gamma_class', 0)
        gamma_bc = params.get('gamma_bc', 0)
        gamma_graph = params.get('gamma_graph', 0)
        weight_classes = params.get('weight_classes', None)
        gamma_focal = params.get('gamma_focal', 0)
        loss_function = LossODEProcess(gamma_rec=gamma_rec, 
                                       gamma_lat=gamma_lat, 
                                       gamma_class=gamma_class, 
                                       gamma_bc=gamma_bc, 
                                       gamma_graph=gamma_graph,
                                       weight_classes=weight_classes, gamma_focal=gamma_focal, use_mse=self.use_mse)
        # loss_function = nn.MSELoss(reduction='none')

        # =================================================================================================
        # OPTIMIZER HERE
        # =================================================================================================        
        # Optimizer
        # Adam / AdamW / RMSprop / Rprop / ASGD
        # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay, amsgrad=False, eps=1e-8, betas=(0.9, 0.999))
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=init_lr, alpha=0.99)

        # Scheduler
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5, last_epoch=-1)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                                         factor=params.get('lr_reduce_factor', 0.4),
        #                                                         patience=params.get('lr_schedule_patience', 10),
        #                                                         threshold=params.get('lr_schedule_threshold', 1e-4),
        #                                                         min_lr=params.get('lr_min', 1e-6),
        #                                                         )
        
        # Scheduler with early cut-off factor of 1.15
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int((num_epochs-100) * 1.15), eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)

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
        pin_memory = True if self.device == 'cuda' else False
        train_dataloader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=drop_last, sampler=None,
                                      prefetch_factor=4, num_workers=num_workers_per_dataloader, pin_memory=pin_memory, persistent_workers=True)
        if valid_dataset is not None:
            val_dataloader = DataLoader(valid_dataset, collate_fn=collate, batch_size=len(valid_dataset))
        else:
            val_dataloader = None

        if test_dataset is not None:
            test_dataloader = DataLoader(test_dataset, collate_fn=collate, batch_size=len(test_dataset))    
        else:
            test_dataloader = None        

        # Train the model 
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
                                       gamma_bc=params['gamma_bc'],
                                       gamma_class=params['gamma_class'],
                                       gamma_graph=params['gamma_graph'],
                                       use_position=params.get('use_position', False),
                                       use_region=params.get('use_region', False),
                                       error_score=self.error_score,
                                       weight_classes=params.get('weight_classes', None),
                                       gamma_focal=params.get('gamma_focal', 1),
                                       multiplex=False,
                                       save_model=save_model,
                                       model_to_load=model_to_load,
                                       fine_tune_model=fine_tune_model,
                                       warmup_epochs=100,
                                       )
            logging.info("Model trained!\n")
        except ValueError as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")
            res_training = {'best_score': self.error_score}

        return res_training