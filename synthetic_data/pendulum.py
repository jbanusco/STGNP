import argparse
import numpy as np
import os
import shutil
import dgl
import pandas as pd
import torch
import json
import multiprocessing
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from synthetic_data.synthetic_dataset import SyntheticDataset
from scipy.spatial.distance import pdist, squareform

from model.train_stmgcn_ode import build_model_multiplex
from synthetic_data.train_synthetic_models import ObjectiveSynthetic, batch_loop
from synthetic_data.gene_evolution import generate_graph
from utils.model_selection_sklearn import stratified_split
from utils.utils import seed_everything, str2bool, get_best_params
from experiments.ACDC_CV import plot_results, plot_combined_trajectories, get_data_in_original_scale
from experiments.ACDC_All import plot_predicted_trajectories
import torchvision.transforms as transforms
from utils.model_selection_optuna import hypertune_optuna
from model.testing_model import get_latex_table, wrap_latex_table, save_training_convergence


def similarity_exp(dist, gamma=1, threshold=None):
    if threshold is not None:
        return np.exp(-np.maximum(dist - threshold, 0) / gamma)
    return np.exp(-dist / gamma)



def generate_data(ode_system, 
                  num_samples=500,
                  length=10,
                  dt=0.1,
                  num_nodes=2,
                  space_coupling=0.1,
                  time_coupling=0.1,
                  init_sample=0,
                  save_folder='data',
                  predicted_length=10,
                  ):
    
    # ==================== Initial states ================
    # Get the state range of the system
    # data_range = np.stack(ode_system.state_range, axis=0)
    data_range = np.stack(ode_system.p1.state_range, axis=0) / 2

    # Separate the minimum and maximum values for each dimension
    min_vals = data_range[:, 0]
    max_vals = data_range[:, 1]

    # Generate the random initial conditions in the range [0, 1]
    # Now, special case, here ndim is the dimension of each pendulum, or 'node' in the system
    ndim = ode_system.p1.ndim
    random_values = np.random.rand(num_samples, num_nodes, ndim)

    # Scale each dimension by the range (min, max)
    initial_states = min_vals + random_values * (max_vals - min_vals)

    # ==================== Adjacency matrices ================
    # These ones will be the same for all subjects - for the moment
    # Edge features = 'coupling' strength or proximity
    
    # This is what is going to go to the model
    A_m = generate_graph(num_nodes, 'fully_connected', {})
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

    sample_ids = np.arange(init_sample, init_sample+num_samples)
    for i, sample_id in enumerate(sample_ids):
        xyz[i] = ode_system.simulate(length=length, dt=dt, init_state=initial_states[i])
        xyz_predicted[i] = ode_system.simulate(length=predicted_length+dt*1.1, dt=dt, init_state=xyz[i, -1])[1:]

        # # Plot the trajectories of the pendulum
        # xyz_plot = np.concatenate([xyz[i], xyz_predicted[i]], axis=0)
        # t_plot = np.arange(0, len(xyz_plot), 1)
        # import matplotlib
        # matplotlib.use('TkAgg')
        # fig, ax  = plt.subplots(1, 2, figsize=(6, 6))
        # ax[0].plot(t_plot, xyz_plot[:, 0, 0], label='Pendulum 1')
        # ax[0].plot(t_plot, xyz_plot[:, 1, 0], label='Pendulum 2')
        # ax[0].set_title('Angle')
        # ax[0].set_xlabel('Time')
        # ax[0].set_ylabel('Angle')
        # ax[1].plot(t_plot, xyz_plot[:, 0, 1], label='Pendulum 1')
        # ax[1].plot(t_plot, xyz_plot[:, 1, 1], label='Pendulum 2')
        # ax[1].set_title('Angular velocity')
        # ax[1].set_xlabel('Time')
        # plt.show()


        # Let's see if we can generate the graphs for DGL
        graph_dict = {}
            
        # Create the 'spatial' graph
        u_space, v_space = np.nonzero(A_m)  # Symmetric, non-directed
        graph_dict[('region', 'space', 'region')] = (u_space, v_space)

        # Create the 'temporal' graph
        u_time, v_time = np.nonzero(At_m)  # Identity matrix
        graph_dict[('region', 'time', 'region')] = (u_time, v_time)

        hg_graph = dgl.heterograph(graph_dict)

        # Assign the edge data
        # Add dimension for the 'number of edges' and for the 'time' dimension
        # hg_graph.edges['space'].data['cat'] = torch.tensor(W[i, u_space, v_space]).unsqueeze(1).unsqueeze(2).repeat(1, 1, len(t))
        hg_graph.edges['space'].data['cat'] = torch.tensor(W[u_space, v_space]).unsqueeze(1).unsqueeze(2).repeat(1, 1, len(t))
        hg_graph.edges['time'].data['cat'] = torch.tensor(Wt[u_time, v_time]).unsqueeze(1).unsqueeze(2).repeat(1, 1, len(t))

        # Assign the node data
        region_id_data = torch.arange(num_nodes)
        region_id_data = pd.get_dummies(region_id_data)

        # ATTENTION!: HERE IS WHERE WE NEED TO ADD THE FEATURES
        # [Batch, Time, Nodes, Features] -> [ Batch, Nodes, Features, Time]
        hg_graph.nodes['region'].data['nfeatures'] = torch.tensor(xyz[i]).permute(1, 2, 0)
        hg_graph.nodes['region'].data['nfeatures_predicted'] = torch.tensor(xyz_predicted[i]).permute(1, 2, 0)
        # hg_graph.nodes['region'].data['pos'] = []  # No positions
        hg_graph.nodes['region'].data['time'] = torch.tensor(t).unsqueeze(0).unsqueeze(1).repeat(num_nodes, 1, 1)
        hg_graph.nodes['region'].data['region_id'] = torch.tensor(region_id_data.values)

        # Store the graphs
        dgl.save_graphs(f'{save_folder}/graph_{sample_id}.bin', hg_graph) # Save the graph


class Pendulum(object):
    def __init__(self, mass=1, length=1, damping=0):
        """
        Simple pendulum system. Defined by theta and theta_dot, where theta is the angle with the vertical and
        theta_dot is the angular velocity.
        params:
            mass: mass of the pendulum [kg]
            length: length of the pendulum [m]
            damping: damping coefficient
        """
        self.ndim = 2  # Number of state dimensions [θ, θ_dot]
        self.state_range = [(-np.pi/2, np.pi/2), (-1.0, 1.0)]

        self.g = 9.81  # Gravitational acceleration
        self.m = mass  
        self.l = length
        self.mu = damping

    def system_fn(self, current_state, t):
        θ_ddot = ((-self.mu) * current_state[1]) + (-self.g / self.l) * np.sin(current_state[0])
        θ_dot = current_state[1]

        return [θ_dot, θ_ddot]
    
    def simulate(self, length=50, dt=0.05, init_state=np.array([0, 1])):
        t = np.linspace(0, length, int(length/dt))
        return odeint(self.system_fn, init_state, t)


class CoupledPendulum(object):
    def __init__(self, num_pendulum=2, k=2, m1=1, m2=1, l1=1.5, l2=1.5):
        """
        Coupled pendulum system. Linked by a spring with spring constant k.
        params:
            num_pendulum: number of pendulums
            k: spring constant
            m1: mass of the first pendulum [kg]
            m2: mass of the second pendulum [kg]
            l1: length of the first pendulum [m]
            l2: length of the second pendulum [m]
        """
        self.num_nodes = num_pendulum  # Number of nodes
        self.ndim = num_pendulum * 2  # Number of dimensions
        self.state_range = [(-np.pi/2, np.pi/2), (-1, 1), (-np.pi/2, np.pi/2), (-1, 1)]  # For the coupled

        self.g = 9.81  # Gravitational acceleration
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2

        # Initialized the pendulums
        self.p1 = Pendulum(mass=self.m1, length=self.l1, damping=0)
        self.p2 = Pendulum(mass=self.m2, length=self.l2, damping=0)

    @staticmethod
    def _to_array(state_matrix):
        return state_matrix.flatten()
    
    @staticmethod
    def _to_matrix(state_array, ndim):
        return state_array.reshape(-1, ndim)        
    
    def system_fn(self, current_state, t):
        # Compute acceleration for each pendulum
        θ1_ddot = (np.sin(current_state[0]) * (self.p1.m * (self.p1.l * (current_state[1] * current_state[1]) - self.g) - (self.k * self.p1.l)) + (self.k * self.p2.l * np.sin(current_state[2]))) / (self.p1.m * self.p1.l * np.cos(current_state[0]))
        θ2_ddot = (np.sin(current_state[2]) * (self.p2.m * (self.p2.l * (current_state[3] * current_state[3]) - self.g) - (self.k * self.p2.l)) + (self.k * self.p1.l * np.sin(current_state[0]))) / (self.p2.m * self.p2.l * np.cos(current_state[2]))

        # Current velocities
        θ1_dot = current_state[1]
        θ2_dot = current_state[3]

        return [θ1_dot, θ1_ddot, θ2_dot, θ2_ddot]

    def simulate(self, length=10, dt=0.1, init_state=np.array(([[0, 1], [1, 0]]))):
        input_state = self._to_array(init_state)
        t = np.linspace(0, length, int(length/dt))
        trajectories = odeint(self.system_fn, input_state, t)

        ndim = int(self.ndim / self.num_nodes)  # Remember , for each pendlum we have 2 dimensions
        trajectories = trajectories.reshape(-1, self.num_nodes, ndim)

        return trajectories


class PendulumDataset(SyntheticDataset):
    """
    Coupled pendulum dataset
    """
    def __init__(self, name, save_dir=None, reprocess=False, 
                 num_samples=500, num_nodes=2, spring_constant=2,
                 space_coupling=1, time_coupling=1, 
                 dt=0.1, duration=10, has_labels=False, save_info=True):
        self._name = f"{name}"
        self.has_labels = has_labels
        self.save_info = save_info
        self._save_path = os.path.join(save_dir, self.name)   
        if not os.path.isdir(self._save_path):
            reprocess = True  # For sure, since we don't have the data

        # This dataset has no labels and no additional information
        self.has_labels = has_labels
        self.save_info = save_info        

        # ODE system
        self.system_fn = CoupledPendulum(num_pendulum=2, k=spring_constant)
        self.system_fn2 = CoupledPendulum(num_pendulum=2, k=spring_constant*4)

        # Number of samples
        self.num_samples = num_samples
        self.dt = dt
        self.duration = duration

        # Number of nodes / or 'components' of the system
        self.num_nodes = num_nodes

        # Joint parameters
        self.space_coupling = space_coupling  # Coupling strength in space
        self.time_coupling = time_coupling  # Coupling strength in time

        # Name of the features [for plotting]
        self.features = ['$\\theta$', '$\\dot{\\theta}$']

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
        elif reprocess and os.path.isdir(self._save_path):
            shutil.rmtree(self._save_path)

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
                      init_sample=0,
                      save_folder=self._save_path,
                      predicted_length=self.duration
                      )
        
        # Second group
        # generate_data(self.system_fn2, 
        #               num_samples=self.num_samples,
        #               length=self.duration,
        #               dt=self.dt,
        #               num_nodes=self.num_nodes,
        #               space_coupling=self.space_coupling,
        #               time_coupling=self.time_coupling,
        #               init_sample=self.num_samples,
        #               save_folder=self._save_path,)

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
        num_context = int(0.2*num_frames)   # These are like 'control' points        
        # Get the set of context points
        # num_subjects = in_data.shape[0]
        # context_pts = np.zeros((num_subjects, num_context+1), dtype=int)
        # for s in range(0, num_subjects):
        #     s_context_pts = np.random.choice(np.arange(1, total_points-1), num_context, replace=False)
        #     s_context_pts = np.concatenate([np.array([0]), s_context_pts])  # Add always intial point to the context
        #     context_pts[s, :] = s_context_pts

        # context_pts = np.random.choice(np.arange(1, num_frames-1), num_context, replace=False)
        # context_pts = np.concatenate([np.array([0]), context_pts])  # Add always intial point to the context        
        context_pts = np.arange(0, num_frames+1, 3)  # Fix context points
        target_pts = np.arange(0, num_frames)  # All of them

        # ==================
        # Transform the data
        # ==================
        # The node data is in [Nodes, Features, Time] format
        nfeatures = self._transform['nfeatures'](node_data['nfeatures'].permute(2, 0, 1)).permute(1, 2, 0)
        nfeatures_predicted = self._transform['nfeatures'](node_data['nfeatures_predicted'].permute(2, 0, 1)).permute(1, 2, 0)
        # pos_data = self._transform['pos'](node_data['pos'].permute(2, 0, 1)).permute(1, 2, 0)

        # Transform the edge data -- Don't transform the edges
        edge_space = edge_data['space']
        edge_time = edge_data['time']
        # edge_space = self._transform['space'](edge_data['space'].permute(2, 0, 1)).permute(1, 2, 0)
        # edge_time = self._transform['time'](edge_data['time'].permute(2, 0, 1)).permute(1, 2, 0)

        # Input data
        input_node_data = {'features': nfeatures,
                           # 'pos': pos_data,
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
    # By default
    paper_folder = "/media/jaume/DATA/Data/Multiplex_Synthetic_FINAL"
    # paper_folder = "/usr/data/Multiplex_Synthetic_FINAL"
    # study_name = "Multiplex_CoupledPendulum"
    # study_name = "Multiplex_CoupledPendulum_ADAM"
    # study_name = "Multiplex_CoupledPendulum_ADAM_FINAL_MAE"
    # study_name = "Multiplex_CoupledPendulum_ADAM_END"
    # study_name = "Multiplex_CoupledPendulum_Pred_ADAM"
    # study_name = "Multiplex_CoupledPendulum_Pred"
    # study_name = "Multiplex_CoupledPendulum_DIMENSIONS"
    study_name = "Multiplex_CoupledPendulum_DIMENSIONS_NEW_LOSS"

    parser = argparse.ArgumentParser(description='Coupled Pendulum')

    # Folder
    parser.add_argument('--save_folder', type=str, default=f"{paper_folder}", help='Folder to save the data')
    parser.add_argument('--name', type=str, default='CoupledPendulum', help='Name of the dataset')
    parser.add_argument('--reprocess', type=str2bool, default=False, help='Reprocess the data.')
    parser.add_argument('--reload_model', type=str2bool, default=True, help='Reload the model.')
    parser.add_argument('--experiment_name', type=str, required=False, default=f'{study_name}', help='Name of the experiment.')
    parser.add_argument('--run_best', type=str2bool, default=True, help='Run the best model.')
    parser.add_argument('--is_optuna', type=str2bool, default=False, help='Run the best model.')

    # Simulation
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--k', type=float, default=2, help='Spring constant')
    parser.add_argument('--space_coupling', type=float, default=1, help='Space coupling')
    parser.add_argument('--time_coupling', type=float, default=1, help='Time coupling')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--duration', type=float, default=10, help='Duration of the simulation')

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
    parser.add_argument('--use_bias_stmgcn', type=str2bool, default=False, help='Use bias in the ST-MGCN')
    parser.add_argument('--decode_just_latent', type=str2bool, default=True, help='Decoder just uses the latent space')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials for the hyper-parameter tuning')
    parser.add_argument('--num_jobs', type=int, default=0, help='Number of jobs in parallel')

    parser.add_argument('--gamma_rec', type=float, default=2, help='Weight for the regression loss')
    parser.add_argument('--gamma_class', type=float, default=0., help='Weight for the classification loss')
    parser.add_argument('--gamma_bc', type=float, default=0., help='Weight for the boundary condition in the latent space')
    parser.add_argument('--gamma_lat', type=float, default=0.1, help='L2 weight for the latent space')
    parser.add_argument('--gamma_graph', type=float, default=0.1, help='Weight for the graph regularization')

    return parser


def main():
    # Seed everything
    seed_everything()

    # Parse the arguments
    parser = get_parser()
    args = parser.parse_args()

    run_best = args.run_best
    study_name = args.experiment_name
    duration = float(args.duration)
    dt = float(args.dt)
    k = float(args.k)
    dataset_name = f'duration-{duration}_dt-{dt}_k-{k}'

    # Save folder
    save_folder = os.path.join(args.save_folder, "CoupledPendulum_Graphs", f"{dataset_name}")
    os.makedirs(save_folder, exist_ok=True)

    # Store .json with the configuration
    json_filename = os.path.join(save_folder, 'config.json')
    with open(json_filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Options
    normalization = args.normalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = "cpu"

    # coupled_pendulum = CoupledPendulum(num_pendulum=2, k=2)
    # generate_data(coupled_pendulum, num_samples=500, length=10,
    #               dt=0.1,num_nodes=2, space_coupling=0.1, time_coupling=0.1, save_folder=save_folder)
    
    is_optuna = args.is_optuna
    # if is_optuna:
    #     num_samples = 500   # Just for the hyper-parameter tuning, otherwise takes too long.
    # else:
    num_samples = args.num_samples

    # Create the dataset
    dataset = PendulumDataset(name=args.name,
                              save_dir=save_folder, 
                              reprocess=args.reprocess,
                              space_coupling=args.space_coupling, 
                              time_coupling=args.time_coupling, 
                              dt=dt, 
                              duration=duration, 
                              spring_constant=k,
                              num_samples=num_samples,
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
    gamma_class = args.gamma_class  # Classification
    gamma_lat = args.gamma_lat  # Latent space
    gamma_bc = args.gamma_bc  # Boundary condition
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
                      'gamma_class': gamma_class,
                      'gamma_lat': gamma_lat,
                      'gamma_bc': gamma_bc,
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
                                          class_dim=0,
                                          space_planes=args.space_planes,
                                          time_planes=args.time_planes,
                                          depth_nodes=1,
                                          depth_edges=1,
                                          use_edges=use_edges,
                                          only_spatial=False,
                                          use_norm=args.use_norm,
                                          use_mse=False,
                                          )
    objective_optuna.set_default_params(default_config)
    
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

    sq_database = False  # sqlite or postgresql

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

        results_hp_folder = os.path.join(args.save_folder, "CoupledPendulum_Graphs", "results")
        df_params_path = os.path.join(results_hp_folder, f'{study_name}_trials.csv')
        df_params = pd.read_csv(df_params_path)
        df_params.dropna(how='any', inplace=True)
        df_params = df_params.sort_values(by='value', ascending=False)
        # params_names = [key for key in df_params.columns if key.startswith('params_')]
        # best_params = df_params.iloc[0].to_dict()   
        # # best_params = df_params.iloc[0:5][params_names].mean().to_dict()
        # best_params = {key.replace('params_', ''): value for key, value in best_params.items() if key.startswith('params_')}
        best_params = get_best_params(df_params.iloc[0:5], use_median=True)
        # --- VERY VERY GOOD + depth 1
        best_params['hidden_dim'] = 17
        # best_params['latent_dim'] = 6

        # best_params = {'gamma_lat': 0.37, 'gamma_rec': 0.93, 'gamma_graph': 0.04, 'weight_decay': 0.00017}
        print(f"Best parameters: {best_params}")

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
        study_model_copy = os.path.join(save_folder, f'model_{study_name}.pt')
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
            print(f"Avaliable CPUs: {multiprocessing.cpu_count()}, Using {num_cpus} CPUs, Jobs: {objective_optuna.num_jobs}")
            model, res_training, best_params = hypertune_optuna(objective_optuna,
                                                                save_folder,
                                                                study_name,
                                                                num_trials=num_trials,
                                                                load_previous=load_previous,
                                                                output_probs=True, 
                                                                max_cpus=num_cpus,
                                                                sq_database=sq_database,)
        else:
            # Model
            tmp_save = os.path.join(save_folder, 'TmpModel')
            if not args.reload_model and os.path.isdir(tmp_save):
                shutil.rmtree(tmp_save, ignore_errors=True)
            os.makedirs(tmp_save, exist_ok=True)

            print(f"Params: {objective_optuna.default_params}")
            model_params = objective_optuna.default_params.copy()
            model = objective_optuna.build_model(model_params)

            # Store the parameters
            json_filename = os.path.join(save_folder, 'objective_params.json')
            with open(json_filename, 'w') as f:
                json.dump(model_params, f, indent=4)

            res_training = objective_optuna._train(model, model_params, tmp_save, final_model=True)

    if sq_database or run_best or not is_optuna:
        steps_to_predict = int(duration/dt)
        # steps_to_predict = int(duration / dt_step_size)
        # print(steps_to_predict)
        # time_to_predict = np.arange(0, steps_to_predict, 1)  # Predict 100 steps more
        time_to_predict = torch.arange(0, steps_to_predict, 1)
        pred_trajectory, pred_latent, tgt_trajectory = objective_optuna.predict_from_latent(model, objective_optuna.dataset, time_to_predict, model_params, device=device)
        # The shape of the results is [num_samples, num_features, num_nodes, num_time_steps]

        
        # Save as metric the error in the predicted trajectory
        total_mse = (tgt_trajectory - pred_trajectory.mean[..., 1:]).square().mean(dim=0).sum()
        df_mse = pd.DataFrame({'MSE': [total_mse.item()]})
        df_mse.to_csv(os.path.join(save_folder, 'mse.csv'))
        
        # Per feature
        df_errors = get_data_in_original_scale(model, objective_optuna, model_params, save_folder, pred_trajectory, 
                                               fts_to_predict=None, true_trajectory=tgt_trajectory, normalization=normalization)
        df_errors.to_csv(os.path.join(save_folder, 'errors_per_feature.csv'))
        
        latex_table = get_latex_table(df_errors, objective_optuna)
        latex_table = latex_table.replace('train', 'train window')
        latex_table= latex_table.replace('predict', 'extrapolation')
        wrapped_table = wrap_latex_table(latex_table, caption="Errors per feature in the test set", label="tab:feature_errors_pendulum")
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
                                   fts_to_predict=None, true_trajectory=tgt_trajectory, normalization=normalization, save_format=save_format)
        plot_predicted_trajectories(objective_optuna, pred_latent, pred_trajectory, save_folder, 
                                    normalization=normalization, plot_individual=True, true_trajectory=tgt_trajectory, 
                                    plot_spatial=False, save_format=save_format)
        save_training_convergence(res_training, save_folder, save_format=save_format)
        plot_results(model, objective_optuna, model_params, save_folder, plot_individual=True, plot_spatial=False, save_format=save_format)


if __name__ == '__main__':
    main()