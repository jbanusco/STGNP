import argparse
import json
import numpy as np
import os
import dgl
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from synthetic_data.synthetic_dataset import SyntheticDataset
import shutil
import multiprocessing
import torchvision.transforms as transforms

from synthetic_data.train_synthetic_models import ObjectiveSynthetic, batch_loop
from utils.utils import seed_everything, str2bool, get_best_params
from utils.model_selection_optuna import hypertune_optuna
from utils.model_selection_sklearn import stratified_split
from utils.graph_utils import generate_graph
from model.plot_and_print_utils import get_latex_table, wrap_latex_table, save_training_convergence, plot_results, plot_combined_trajectories, get_data_in_original_scale, plot_predicted_trajectories


def transform_traj(x):    
    trajectories_wrapped = np.sin(x)

    return trajectories_wrapped


class Kuramoto:
    def __init__(self, 
                 num_genes=10,
                 k=1/3,
                 g=1,
                 state_range=(0, 2*np.pi), 
                 A=None):
        self.ndim = 1
        self.num_genes = num_genes
        self.g = g  # Global coupling
        self.k = k  # Edge coupling
        self.state_range = state_range

        self.min_freq = 0.1 # Hz
        self.max_freq = 0.5 # Hz
        self.w = self._init_frequencies()

        # Coupling matrix: Fully connected graph without self-loops by default
        self.A = generate_graph(self.num_genes, 'fully_connected', {})
        self.At = generate_graph(self.num_genes, 'identity', {})        

    def _init_frequencies(self, num_samples=1, homogeneous=True):
        """
        Initialize natural frequencies from a Gaussian distribution within the range of 0.1 to 1 Hz.
        """
        # Convert desired Hz range to rad/s
        omega_min = 2 * np.pi * self.min_freq 
        omega_max = 2 * np.pi * self.max_freq

        # Set Gaussian parameters
        omega_mean = (omega_min + omega_max) / 2  
        omega_std = (omega_max - omega_min) / 6   # Standard deviation (adjust for ~99.7% within range)

        # Sample from Gaussian
        if homogeneous:
            w = np.random.normal(loc=omega_mean, scale=omega_std, size=(num_samples, 1))
            w = np.repeat(w, self.num_genes, axis=1)
        else:
            w = np.random.normal(loc=omega_mean, scale=omega_std, size=(num_samples, self.num_genes))
        
        # Clip to ensure values stay within [omega_min, omega_max]
        w = np.clip(w, omega_min, omega_max)
        
        return w

    def _init_state(self, num_samples=1):
        """
        Initialize the state as a random matrix in the range state_range. Wrap the values to [0, 2*pi].
        """
        state =  np.random.uniform(self.state_range[0], self.state_range[1], size=(num_samples, self.num_genes))
        
        return state


    @staticmethod
    def _to_array(state_matrix):
        """
        Flatten the matrix to a 1D array.
        """
        return state_matrix.flatten()

    @staticmethod
    def _to_matrix(state_array, ndim):
        """
        Reshape the 1D array back to a 2D matrix.
        """
        return state_array.reshape(-1, ndim)

    def system_fn(self, current_state, t):
        """
        Defines the Kuramoto dynamics using vertex_params and edge_params.
        dx/dt = g * sum(A_ij * sin(x_j - x_i)) + b * sin(h - x_i)
        """
        frequencies = current_state[self.num_genes:]
        phases = current_state[:self.num_genes]

        # Reshape the state vector to a matrix
        state_matrix = self._to_matrix(phases, self.ndim)  # Shape: (num_genes, ndim)

        # Compute the pairwise phase differences
        phase_diff = state_matrix[:, None, :] - state_matrix[None, :, :]  # Shape: (num_genes, num_genes, ndim)

        # Compute the coupling term for all nodes simultaneously
        coupling = np.sum(self.A[:, :, None] * np.sin(phase_diff), axis=1)  # Shape: (num_genes, ndim)
        dx = self.g * coupling
        
        # Add the natural frequencies
        dx = dx + frequencies[:, None]  # Shape: (num_samples, num_genes, ndim)

        # Return the derivative as a flattened array
        return np.concatenate((self._to_array(dx), np.zeros_like(frequencies)), axis=0)

    def simulate(self, length=50, dt=0.05, init_state=None, A=None, idx=0):
        """
        Simulate the system for a given time duration.
        """
        if init_state is None:
            init_state = self.init_state
        
        A = self.A if A is None else A
        self.A = A
        
        # Time points
        t = np.linspace(0, length, int(length / dt))
        input_state = self._to_array(init_state)
        input_state = np.concatenate((input_state, self.w[idx,]), axis=0)  # Add the frequencies to the state
        # input_state = np.concatenate((input_state, self.w[0,]), axis=0)  # Homogeneous frequencies
        trajectories = odeint(self.system_fn, input_state, t)
        trajectories = trajectories[:, :self.num_genes].reshape(-1, self.num_genes, self.ndim)

        return trajectories


# Function to wrap phase differences between [-pi, pi]
def wrapped_diff(phase_diff):
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi


# Function to compute the phase difference matrix for a given time t
def compute_phase_diff_matrix(phases):
    """
    Computes pairwise phase differences for each sample at time t_index.
    
    Args:
    phases (ndarray): Shape [Sample, T, Nodes], where `phases[sample, time, node]` gives the phase of the node at a given time.
    t_index (int): The time index at which to compute the phase differences.
    
    Returns:
    phase_diff_matrix (ndarray): Shape [Sample, Nodes, Nodes], where each entry contains the phase difference between two nodes.
    """
    # Extract the phases at the given time for all samples
    # Compute pairwise phase differences using broadcasting
    phase_diff_matrix = phases[:, None, :, :] - phases[:, :, None, :]
    
    # Wrap the phase differences to the range [-pi, pi]
    phase_diff_matrix = wrapped_diff(phase_diff_matrix)
    
    return phase_diff_matrix


def generate_data(ode_system, 
                  num_samples=1,
                  length=50, 
                  dt=0.05,
                  num_nodes=10,
                  space_coupling=0.1, 
                  time_coupling=0.1,
                  save_folder='data',
                  spatial_graph_type='fully_connected',
                  temporal_graph_type='identity',
                  graph_params={},
                  predicted_length=50):
    """
    Generate synthetic graph data for Kuramoto-like dynamics.

    Parameters:
    - ode_system: Kuramoto system function to simulate dynamics.
    - num_samples: Number of graph samples to generate.
    - length: Total duration of simulation.
    - dt: Time step for the simulation.
    - num_nodes: Number of nodes in each graph.
    - space_coupling: Coupling strength for spatial edges.
    - time_coupling: Coupling strength for temporal edges.
    - save_folder: Directory to save generated graphs.
    - spatial_graph_type: Type of spatial graph ('fully_connected', 'grid', 'random', 
                          'small_world', 'barabasi', 'erdos').
    - temporal_graph_type: Type of temporal graph ('identity', 'cyclic', 'fully_connected').
    - graph_params: Dictionary of additional parameters for graph generation
                    (e.g., for 'small_world': {'k': 4, 'p': 0.1}).
    """    
    os.makedirs(save_folder, exist_ok=True)

    # ==================== Initial states ================
    ndim = ode_system.ndim

    # Generate the random initial conditions
    initial_states = ode_system._init_state(num_samples)

    # Adjacency matrix for space (node-to-node interaction)
    ode_system.w = ode_system._init_frequencies(num_samples, homogeneous=True)  # Update the frequencies = initial states

    # ==================== Adjacency matrices ================
    # This is what is going to go to the model
    A_m = generate_graph(num_nodes, 'fully_connected', {})
    # At_m = generate_graph(num_nodes, 'fully_connected', {})
    At_m = generate_graph(num_nodes, 'identity', {})

    # Coupling matrices for spatial and temporal edges
    W = A_m * space_coupling
    Wt = At_m * time_coupling

    # ==================== Generate trajectories ================
    num_steps = int(length/dt)
    t = np.linspace(0, length, num_steps)    
    xyz = np.zeros((num_samples, len(t), num_nodes, ndim))  # [ Batch, Time, Nodes, Features]

    num_steps_predicted = int(predicted_length/dt)
    t_predicted = np.linspace(length, length+predicted_length, num_steps_predicted)
    xyz_predicted = np.zeros((num_samples, len(t_predicted), num_nodes, ndim))  # [ Batch, Time, Nodes, Features]

    # Generate the adjacency matrices -- all samples will have the same underlying graph
    A = generate_graph(num_nodes, spatial_graph_type, graph_params)
    At = generate_graph(num_nodes, temporal_graph_type, graph_params)

    # Update A - Graph connectivity used to simulate the system
    ode_system.A = A * ode_system.k
    ode_system.At = At
    
    for i in range(num_samples):
        xyz[i] = ode_system.simulate(length=length, dt=dt, init_state=initial_states[i], A=None, idx=i)
        xyz_predicted[i] = ode_system.simulate(length=predicted_length+dt*1.1, dt=dt, init_state=xyz[i, -1], A=None, idx=i)[1:]

        # Create graphs with DGL
        graph_dict = {}

        # Spatial graph edges
        u_space, v_space = np.nonzero(A_m)  # Symmetric, non-directed
        graph_dict[('region', 'space', 'region')] = (u_space, v_space)

        # Temporal graph edges
        u_time, v_time = np.nonzero(At_m)  # Identity matrix
        graph_dict[('region', 'time', 'region')] = (u_time, v_time)

        hg_graph = dgl.heterograph(graph_dict)

        # Add edge data for spatial and temporal connections
        hg_graph.edges['space'].data['cat'] = torch.tensor(W[u_space, v_space]).unsqueeze(1).unsqueeze(2).repeat(1, 1, len(t))
        hg_graph.edges['time'].data['cat'] = torch.tensor(Wt[u_time, v_time]).unsqueeze(1).unsqueeze(2).repeat(1, 1, len(t))

        # Add node data
        region_id_data = torch.arange(num_nodes)
        region_id_data = pd.get_dummies(region_id_data)

        hg_graph.nodes['region'].data['nfeatures'] = torch.tensor(xyz[i]).permute(1, 2, 0)
        hg_graph.nodes['region'].data['nfeatures_predicted'] = torch.tensor(xyz_predicted[i]).permute(1, 2, 0).float()
        hg_graph.nodes['region'].data['time'] = torch.tensor(t).unsqueeze(0).unsqueeze(1).repeat(num_nodes, 1, 1)
        hg_graph.nodes['region'].data['region_id'] = torch.tensor(region_id_data.values)

        # Save graph
        dgl.save_graphs(f'{save_folder}/graph_{i}.bin', hg_graph)



class KuramotoDataset(SyntheticDataset):
    """
    Kuramoto dataset
    """
    def __init__(self, name, save_dir=None, reprocess=False,
                 has_labels=False, save_info=True,
                 num_samples=10,
                 num_nodes=3,
                 duration=50,
                 dt=0.05,
                 spatial_coupling=1,
                 temporal_coupling=1,
                 k=1/3,
                 g=1,
                 graph_params={},
                 spatial_graph_type='barabasi', temporal_graph_type='identity'):
        
        self._name = f"{name}"
        self.has_labels = has_labels
        self.save_info = save_info
        self._save_path = os.path.join(save_dir, self.name)   
        if not os.path.isdir(self._save_path):
            reprocess = True  # For sure, since we don't have the data
        else:
            if reprocess:
                shutil.rmtree(self._save_path, ignore_errors=True)
                os.makedirs(self._save_path, exist_ok=True)
        # ODE system
        self.system_fn = Kuramoto(num_genes=num_nodes, k=k, g=g)
        self.duration = duration
        self.dt = dt

        # Number of samples
        self.num_samples = num_samples

        # Number of nodes
        self.num_nodes = num_nodes
        self.k = k  # Coupling of oscillators

        # Joint parameters -- for the model initialization
        self.space_coupling = spatial_coupling
        self.time_coupling = temporal_coupling

        # Graph parameters
        self.graph_params = graph_params
        self.spatial_graph_type = spatial_graph_type
        self.temporal_graph_type = temporal_graph_type

        # Name of the features [states]
        self.features = ['$x$']

        if not reprocess:
            self.load()
            super().__init__(name, save_dir=save_dir, reprocess=reprocess, has_labels=self.has_labels, save_info=self.save_info)
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
    
        super().__init__(name, save_dir=save_dir, reprocess=reprocess, has_labels=self.has_labels, save_info=self.save_info)
        self._transform = {
            'pos': transforms.Normalize(mean=0, std=1), 
            'nfeatures': transforms.Normalize(mean=0, std=1),
            'global': transforms.Normalize(mean=0, std=1),
            'space': transforms.Normalize(mean=0, std=1),
            'time': transforms.Normalize(mean=0, std=1),
            }
        
    def process(self):
        # Create the folder
        os.makedirs(self._save_path, exist_ok=True)

        # Generate the graphs
        generate_data(self.system_fn, 
                      num_samples=self.num_samples,
                      length=self.duration,
                      dt=self.dt,
                      num_nodes=self.num_nodes,
                      space_coupling=self.space_coupling,
                      time_coupling=self.time_coupling,
                      save_folder=self._save_path,
                      spatial_graph_type=self.spatial_graph_type,
                      temporal_graph_type=self.temporal_graph_type,
                      graph_params=self.graph_params,
                      predicted_length=self.duration,
                      )

        # Load them as a list of graphs
        list_graphs = [g for g in os.listdir(self.save_dir) if g.endswith('.bin') and self.name not in g]
        graph_collection = []
        node_data = []
        edge_data = []

        for graph_filename in list_graphs:
            hg = dgl.load_graphs(os.path.join(self.save_dir, graph_filename))[0][0]
            graph_collection.append(hg)

            ndata = hg.ndata['nfeatures']
            ndata_predicted = hg.ndata['nfeatures_predicted']
            time = hg.ndata['time']
            region_id = hg.ndata['region_id']
            node_data.append({'nfeatures': ndata, 
                              'nfeatures_predicted': ndata_predicted,
                              'time': time, 
                              'pos': [], 
                              'region_id': region_id})

            edata_space = hg['space'].edata['cat']
            edata_time = hg['time'].edata['cat']
            edge_data.append({'space': edata_space, 'time': edata_time})

        # Store the loaded data
        self.graph = graph_collection
        self.nodes_data = node_data
        self.edges_data = edge_data
        self.sub_id = np.arange(len(graph_collection))
        self.global_data = []

        # Get the list of node, edge and global features
        self._node_features_id()
        self._edge_features_id()
        self._global_features_id()


    def __getitem__(self, idx):
        # Get the graph
        graph = self.graph[idx]

        # Nodes and edge data
        node_data = self.nodes_data[idx]
        edge_data = self.edges_data[idx]

        # ==================
        # Get the context points
        # ==================
        num_frames = node_data['nfeatures'].shape[-1]
        context_pts = np.arange(0, num_frames+1, 3)  # Fix context points
        target_pts = np.arange(0, num_frames)  # All of them

        # ==================
        # Transform the data
        # ==================
        nfeatures = self._transform['nfeatures'](node_data['nfeatures'].permute(2, 0, 1)).permute(1, 2, 0)        
        nfeatures_predicted = self._transform['nfeatures'](node_data['nfeatures_predicted'].permute(2, 0, 1)).permute(1, 2, 0)

        # Transform the edge data -- Don't transform the edges
        edge_space = edge_data['space']
        edge_time = edge_data['time']

        # Input data
        input_node_data = {'features': nfeatures,
                           'time': node_data['time'],
                           'region_id': node_data['region_id'],
                           'features_predicted': nfeatures_predicted,
                           }
        
        input_edge_data = {'space': edge_space,
                           'time': edge_time,
                           }
        
        label = self.labels[idx] if self.has_labels else np.nan

        return graph, label, context_pts, target_pts, input_node_data, input_edge_data


def get_parser():
    paper_folder = "/media/jaume/DATA/Data/Multiplex_Synthetic_FINAL"
    study_name = "Multiplex_Kuramoto_DIMENSIONS_NEW_LOSS"
    
    parser = argparse.ArgumentParser(description='Kuramoto Oscillators')

    # Folder
    parser.add_argument('--save_folder', type=str, default=f"{paper_folder}", help='Folder to save the data')
    parser.add_argument('--name', type=str, default='KuramotoOscillator', help='Name of the dataset')
    parser.add_argument('--reprocess', type=str2bool, default=False, help='Reprocess the data')
    parser.add_argument('--reload_model', type=str2bool, default=True, help='Reload the model')
    parser.add_argument('--experiment_name', type=str, required=False, default=f'{study_name}', help='Name of the experiment.')
    parser.add_argument('--run_best', type=str2bool, default=True, help='Run the best model')
    parser.add_argument('--is_optuna', type=str2bool, default=False, help='Run the best model')
    
    # Simulation
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes. Should be a square number. 16=4x4 grid')
    parser.add_argument('--k', type=float, default=0.33, help='Oscillator coupling')
    parser.add_argument('--space_coupling', type=float, default=1, help='Space coupling')
    parser.add_argument('--time_coupling', type=float, default=1, help='Time coupling')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--duration', type=float, default=10, help='Duration of the simulation')
    parser.add_argument('--spatial_graph_type', type=str, default='barabasi', help='Type of spatial graph') # 'fully_connected', 'identity', 'small_world', 'barabasi', 'erdos'
    parser.add_argument('--temporal_graph_type', type=str, default='identity', help='Type of temporal graph')

    # Model parameters
    parser.add_argument('--normalization', type=str, default='ZNorm', help='Normalization method')  # ZNorm, NoNorm
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=20, help='Hidden dimension')
    parser.add_argument('--space_planes', type=int, default=3, help='Number of space planes')
    parser.add_argument('--time_planes', type=int, default=3, help='Number of time planes')

    parser.add_argument('--use_norm', type=str2bool, default=True, help='Use normalization in the Encoder-Decoder')
    parser.add_argument('--use_diffusion', type=str2bool, default=False, help='Use diffusion in the model')
    parser.add_argument('--use_einsum', type=str2bool, default=False, help='Use einsum in the model')
    parser.add_argument('--agg_type', type=str, default='score', help='Aggregation type')  # sum, mean, flatten
    parser.add_argument('--use_attention', type=str2bool, default=False, help='Use attention in the model')
    parser.add_argument('--compute_derivative', type=str2bool, default=True, help='Compute the derivative')
    parser.add_argument('--use_norm_stmgcn', type=str2bool, default=True, help='Use normalization in the ST-MGCN')
    parser.add_argument('--use_bias_stmgcn', type=str2bool, default=False , help='Use bias in the ST-MGCN')
    parser.add_argument('--decode_just_latent', type=str2bool, default=True, help='Decoder just uses the latent space')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials for the hyper-parameter tuning')
    parser.add_argument('--num_jobs', type=int, default=0, help='Number of jobs in parallel')
    
    parser.add_argument('--gamma_rec', type=float, default=1., help='Weight for the regression loss')
    parser.add_argument('--gamma_lat', type=float, default=0.1, help='L2 weight for the latent space')
    parser.add_argument('--gamma_graph', type=float, default=0.1, help='Weight for the graph regularization')
    
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset')
    
    return parser


def main():
    seed_everything()
    
    # Parse the arguments
    parser = get_parser()
    args = parser.parse_args()

    run_best = args.run_best
    study_name = args.experiment_name
    duration = float(args.duration)
    dt = float(args.dt)
    k = float(args.k)
    num_nodes = int(args.num_nodes)
    spatial_graph_type = args.spatial_graph_type
    temporal_graph_type = args.temporal_graph_type
    if args.dataset_name is None:
        dataset_name = f'duration-{duration}_dt-{dt}_k-{k}_nodes-{num_nodes}_space-{spatial_graph_type}_time-{temporal_graph_type}'
    else:
        dataset_name = args.dataset_name
    print(dataset_name)

    # Discard configurations that are not possible
    if args.use_einsum and args.agg_type != 'sum':
        raise ValueError('Can only use einsum with sum aggregation')
        sys.exit(0)

    if args.use_diffusion and args.agg_type == 'flatten':
        raise ValueError('Cannot use flatten aggregation with diffusion')
        sys.exit(0)

    # Save folder
    save_folder = os.path.join(args.save_folder, "Kuramoto_Graphs", f"{dataset_name}")
    os.makedirs(save_folder, exist_ok=True)

    # Store .json with the configuration
    json_filename = os.path.join(save_folder, 'config.json')
    with open(json_filename, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    normalization = args.normalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    is_optuna = args.is_optuna
    if is_optuna:
        num_samples = 100   # Just for the hyper-parameter tuning, otherwise takes too long.
    else:
        num_samples = args.num_samples

    # Create the dataset        
    dataset = KuramotoDataset(name='KuramotoOscillator',
                              save_dir=save_folder,
                              reprocess=args.reprocess,
                              has_labels=False,
                              save_info=True,
                              spatial_graph_type='barabasi', 
                              temporal_graph_type='identity',
                              g=1,
                              spatial_coupling=args.space_coupling, 
                              temporal_coupling=args.time_coupling,
                              k=k,
                              num_samples=num_samples, 
                              num_nodes=num_nodes, 
                              duration=duration, dt=dt
                              )
    
    # Default model parameters
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    hidden_dim_ext = 0
    dropout = 0. # Not used
    init_lr = args.init_lr
    weight_decay = args.weight_decay
    l1_weight = 0.
    l2_weight = 0.
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    use_attention = args.use_attention
    use_edges = True # Use the edges in the model / predict edges or just use ones
    use_constant_edges = False   # If False take the average of the edges initially and update them? 
    use_hdyn = True
    cond_on_time = False
    decode_just_latent = args.decode_just_latent

    # Loss weights
    gamma_rec = args.gamma_rec  # Regression
    gamma_lat = args.gamma_lat  # Latent space    
    gamma_graph = args.gamma_graph  # Graph regularization
    
    use_region_id = False
    use_position = False
    use_time = False
    dt_step_size = 1/((duration/dt)*0.5)
    
    default_config = {'hidden_dim': hidden_dim, 
                      'hidden_dim_ext': hidden_dim_ext, 
                      'latent_dim': latent_dim,
                      'dropout': dropout,
                      'init_lr': init_lr, 
                      'weight_decay': weight_decay, 
                      'l1_weight': l1_weight, 
                      'l2_weight': l2_weight, 
                      'batch_size': batch_size, 
                      'num_epochs': num_epochs,
                      'use_attention': use_attention,
                      'use_constant_edges': use_constant_edges,
                      'gamma_rec': gamma_rec,
                      'gamma_lat': gamma_lat,
                      'gamma_graph': gamma_graph,
                      'use_region': use_region_id,
                      'decode_just_latent': decode_just_latent,
                      'cond_on_time': cond_on_time,
                      'use_hdyn': use_hdyn,
                      'use_position': use_position,
                      'encode_time': use_time,
                      'only_space': False,
                      'dt_step': dt_step_size,
                      'use_diffusion': args.use_diffusion,
                      'use_einsum': args.use_einsum,
                      'agg_type': args.agg_type,
                      'compute_derivative': args.compute_derivative,
                      'use_norm_stmgcn': args.use_norm_stmgcn,
                      'use_bias_stmgcn': args.use_bias_stmgcn,
                      'use_norm': args.use_norm,
                      }
    
    objective_optuna = ObjectiveSynthetic(study_name,
                                          dataset,
                                          normalization=normalization,
                                          norm_only_ed=False,
                                          save_dir=save_folder,
                                          direction="maximize",
                                          device = device,
                                          track_experiment=False,
                                          use_position=use_position,
                                          use_region_id=use_region_id,
                                          use_time=use_time,
                                          fn_batch_loop=batch_loop,
                                          space_planes=args.space_planes,
                                          time_planes=args.time_planes,
                                          depth_nodes=2,
                                          depth_edges=2,
                                          use_edges=use_edges,
                                          only_spatial=False,
                                          use_norm=args.use_norm,
                                          use_mse=False,
                                          )
    objective_optuna.set_default_params(default_config)
    
    # Store the objective parameters
    json_filename = os.path.join(save_folder, 'objective_params.json')
    with open(json_filename, 'w') as f:
        json.dump(objective_optuna.default_params, f, indent=4)

    # ============================================================================================================================
    # ============================================================================================================================
    # ======================================================== Single run ========================================================
    # Indices for the train / valid split
    indices = np.arange(len(dataset))
    labels = np.ones(len(dataset))  # Dummy labels
    splits = stratified_split(indices, labels, test_size=0.2, valid_size=0.2)
    train_idx = splits['X_train']
    valid_idx = splits['X_valid']
    test_idx = splits['X_test']
    objective_optuna.set_indices(train_idx, valid_idx, test_idx=test_idx)

    sq_database = False

    # Just one run with the default parameters    
    if run_best:
        tmp_save = os.path.join(save_folder, 'FinalModel')
        final_model = os.path.join(tmp_save, 'model.pt')
        study_model_copy = os.path.join(tmp_save, f'model_{study_name}.pt')
        if not args.reload_model and os.path.isdir(tmp_save):
            if os.path.isfile(final_model):            
                os.system(f"rm {final_model}")
            checkpoint_name = os.path.join(tmp_save, 'checkpoint.pt')
            if os.path.isfile(checkpoint_name):
                os.system(f"rm {checkpoint_name}")
        os.makedirs(tmp_save, exist_ok=True)

        results_hp_folder = os.path.join(args.save_folder, "Kuramoto_Graphs", "results")
        df_params_path = os.path.join(results_hp_folder, f'{study_name}_trials.csv')
        df_params = pd.read_csv(df_params_path)
        df_params.dropna(how='any', inplace=True)
        df_params = df_params.sort_values(by='value', ascending=False)
        best_params = get_best_params(df_params.iloc[0:5], use_median=True)
        best_params['hidden_dim'] = 17
        best_params['latent_dim'] = 6
        print(best_params)

        model_params = objective_optuna.default_params.copy()
        model_params.update(best_params)        

        # Store the parameters
        json_filename = os.path.join(save_folder, f'objective_params_{study_name}.json')
        with open(json_filename, 'w') as f:
            json.dump(model_params, f, indent=4)

        # Model
        model = objective_optuna.build_model(model_params)
        res_training = objective_optuna._train(model, model_params, tmp_save, final_model=True)
        os.system(f"cp {final_model} {study_model_copy}")
    else:
        # ============================================================================================================================
        # ============================================================================================================================
        # ====================================== Hyper-parameter tuning on fix train-test split ======================================
        final_model = os.path.join(save_folder, 'model.pt')
        if os.path.isfile(final_model):
            os.system(f"rm {final_model}")
            checkpoint_name = os.path.join(save_folder, 'checkpoint.pt')
            if os.path.isfile(checkpoint_name):
                os.system(f"rm {checkpoint_name}")
        
        if is_optuna:
            num_trials = args.num_trials
            if args.num_jobs > 0:
                num_cpus = args.num_jobs
            else:
                num_cpus = multiprocessing.cpu_count() // 2
            objective_optuna.num_jobs = num_cpus
            load_previous = True
            model, res_training, best_params = hypertune_optuna(objective_optuna,
                                                                save_folder,
                                                                study_name,
                                                                num_trials=num_trials,
                                                                load_previous=load_previous,
                                                                output_probs=True, max_cpus=num_cpus,
                                                                sq_database=sq_database,)
        else:
            # Model
            model_params = objective_optuna.default_params.copy()
            model = objective_optuna.build_model(model_params)
            res_training = objective_optuna._train(model, model_params, tmp_save, final_model=True)
    
    if sq_database or run_best:
        steps_to_predict = int(duration/dt)
        time_to_predict = torch.arange(0, steps_to_predict, 1)
        pred_trajectory, pred_latent, tgt_trajectory = objective_optuna.predict_from_latent(model, objective_optuna.dataset, time_to_predict, model_params, device=device)

        # Per feature
        df_errors = get_data_in_original_scale(model, objective_optuna, model_params, save_folder, pred_trajectory, 
                                               fts_to_predict=None, true_trajectory=tgt_trajectory, 
                                               normalization=normalization, transform=lambda x: x)
        df_errors.to_csv(os.path.join(save_folder, 'errors_per_feature.csv'))
        
        latex_table = get_latex_table(df_errors, objective_optuna)
        latex_table = latex_table.replace('train', 'train window')
        latex_table= latex_table.replace('predict', 'extrapolation')        
        wrapped_table = wrap_latex_table(latex_table, caption="Errors per feature in the test set.", label="tab:feature_errors_kuramoto")
        print(wrapped_table)

        save_format = 'eps'  # svg, png, pdf, eps
        import matplotlib as mpl
        mpl.rcParams.update({
            'axes.titlesize': 28,
            'axes.labelsize': 28,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'legend.fontsize': 10
        })

        plot_combined_trajectories(model, objective_optuna, model_params, save_folder, pred_latent, pred_trajectory, plot_individual=True, plot_spatial=False, 
                                   fts_to_predict=None, true_trajectory=tgt_trajectory, normalization=normalization, save_format=save_format, transform=transform_traj)
        plot_predicted_trajectories(objective_optuna, pred_latent, pred_trajectory, save_folder, 
                                    normalization=normalization, plot_individual=True, true_trajectory=tgt_trajectory, 
                                    plot_spatial=False, save_format=save_format, transform=transform_traj)
        save_training_convergence(res_training, save_folder, save_format=save_format)
        plot_results(model, objective_optuna, model_params, save_folder, plot_individual=True, plot_spatial=False, save_format=save_format, transform=transform_traj)


if __name__ == '__main__':
    main()