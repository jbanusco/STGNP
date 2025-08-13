import argparse
import json
import numpy as np
import os
import dgl
import pandas as pd
import torch
import shutil
import multiprocessing
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from synthetic_data.synthetic_dataset import SyntheticDataset
import torchvision.transforms as transforms


from synthetic_data.train_synthetic_models import ObjectiveSynthetic, batch_loop

from utils.model_selection_sklearn import stratified_split
from utils.utils import seed_everything, str2bool, get_best_params
from utils.model_selection_optuna import hypertune_optuna
from utils.graph_utils import generate_graph
from model.plot_and_print_utils import get_latex_table, wrap_latex_table, save_training_convergence, plot_results, plot_combined_trajectories, get_data_in_original_scale, plot_predicted_trajectories



def plot_components(xyz, t):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
    ax[0].plot(t, x, color='purple', alpha=0.7, linewidth=0.3)
    ax[0].set_title('x component')
    ax[1].plot(t, y, color='purple', alpha=0.7, linewidth=0.3)
    ax[1].set_title('y component')
    ax[2].plot(t, z, color='purple', alpha=0.7, linewidth=0.3)
    ax[2].set_title('z component')
    plt.show()


def plot_system_2d(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # now plot two-dimensional cuts of the three-dimensional phase space
    fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(17, 6))

    # plot the x values vs the y values
    ax[0].plot(x, y, color='purple', alpha=0.7, linewidth=0.3)
    ax[0].set_title('x-y phase plane')

    # plot the x values vs the z values
    ax[1].plot(x, z, color='purple', alpha=0.7, linewidth=0.3)
    ax[1].set_title('x-z phase plane')

    # plot the y values vs the z values
    ax[2].plot(y, z, color='purple', alpha=0.7, linewidth=0.3)
    ax[2].set_title('y-z phase plane')

    # fig.savefig('{}/lorenz-attractor-phase-plane.png'.format(save_folder), 
    #             dpi=180, bbox_inches='tight')
    plt.show()

def plot_system_3d(xyz):
    # extract the individual arrays of x, y, and z values from the array of arrays
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # plot the lorenz attractor in three-dimensional phase space
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((-30, 30))
    ax.set_ylim((-30, 30))
    ax.set_zlim((0, 50))    
    ax.plot(x, y, z, color='purple', alpha=0.7, linewidth=0.3)
    ax.set_title('Lorenz attractor phase diagram')
    plt.show()


def generate_data(ode_system, 
                  num_samples=500,
                  length=10,
                  dt=0.1,
                  num_nodes=2,
                  space_coupling=0.1,
                  time_coupling=0.1,
                  save_folder='data',
                  predicted_length=10,
                  ):
    
    # ==================== Initial states ================
    # Get the state range of the system
    data_range = np.stack(ode_system.state_range, axis=0)

    # Separate the minimum and maximum values for each dimension
    min_vals = data_range[0]
    max_vals = data_range[1]

    # Generate the random initial conditions in the range [0, 1]    
    ndim = ode_system.ndim
    random_values = np.random.rand(num_samples, num_nodes, ndim)

    # Scale each dimension by the range (min, max)
    initial_states = min_vals + random_values * (max_vals - min_vals)

    # ==================== Adjacency matrices ================
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
    
    for i in range(num_samples):
        A = generate_graph(num_nodes, 'fully_connected', {}) # Fully connected graph without self-loops
        At = generate_graph(num_nodes, 'identity', {})

        # Generate H --- For the moment, this won't be passed (too much information)
        H = np.random.randn(num_nodes, num_nodes) * A
        H = H / np.linalg.norm(H, axis=1, keepdims=True)
        H = H * ode_system.k  # Coupling strength

        # Update H - this is what is going to be used to simulate the system
        ode_system.H = H

        xyz[i] = ode_system.simulate(length=length, dt=dt, init_state=initial_states[i])
        xyz_predicted[i] = ode_system.simulate(length=predicted_length+dt*1.1, dt=dt, init_state=xyz[i, -1])[1:]
        
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
        hg_graph.edges['space'].data['cat'] = torch.tensor(W[u_space, v_space]).unsqueeze(1).unsqueeze(2).repeat(1, 1, len(t))
        hg_graph.edges['time'].data['cat'] = torch.tensor(Wt[u_time, v_time]).unsqueeze(1).unsqueeze(2).repeat(1, 1, len(t))

        # Assign the node data
        region_id_data = torch.arange(num_nodes)
        region_id_data = pd.get_dummies(region_id_data)

        # ATTENTION!: HERE IS WHERE WE NEED TO ADD THE FEATURES
        # [ Batch, Time, Nodes, Features] -> [ Batch, Nodes, Features, Time]
        hg_graph.nodes['region'].data['nfeatures'] = torch.tensor(xyz[i]).permute(1, 2, 0)
        hg_graph.nodes['region'].data['nfeatures_predicted'] = torch.tensor(xyz_predicted[i]).permute(1, 2, 0)
        hg_graph.nodes['region'].data['time'] = torch.tensor(t).unsqueeze(0).unsqueeze(1).repeat(num_nodes, 1, 1)
        hg_graph.nodes['region'].data['region_id'] = torch.tensor(region_id_data.values)

        # Store the graphs
        dgl.save_graphs(f'{save_folder}/graph_{i}.bin', hg_graph) # Save the graph



# Define the lorenz system
class Lorenz(object):
    def __init__(self, sigma=10, rho=28, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.state_range = [-1., 1.]
        self.ndim = 3

    def system_fn(self, current_state, t):
        x, y, z = current_state
        xdot = self.sigma * (y - x)
        ydot = x * (self.rho - z) - y
        zdot = x * y - self.beta * z
        return [xdot, ydot, zdot] 

    def simulate(self, length=50, dt=0.015, init_state=np.array([0, 0, 0])):
        t = np.linspace(0, length, int(length/dt))        
        return odeint(self.system_fn, init_state, t)


# Now, the system of coupled Lorenz attractors
class CoupledLorenz(object):
    def __init__(self, num_nodes=4, k=0.01, sigma=10, rho=28, beta=8/3):
        self.state_range = [-1., 1.]
        self.num_nodes = num_nodes
        self.ndim = 3 #* num_nodes
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.k = k  # Coupling strength

        # Spatial matrix
        self.A = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)  # Fully connected graph without self-loops

        # Temporal matrix
        self.At = np.eye(num_nodes)  # Identity matrix

        self.H = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)  # Coupling matrix
        # Normalize the matrix A
        self.H = self.H / np.linalg.norm(self.H, axis=1, keepdims=True)
        # Multiply by the coupling strength
        self.H = self.H * self.k

    @staticmethod
    def _to_array(state_matrix):
        return state_matrix.flatten()
    
    @staticmethod
    def _to_matrix(state_array, ndim):
        return state_array.reshape(-1, ndim)        
    
    def system_fn(self, current_state, t):
        state_matrix = self._to_matrix(current_state, self.ndim)
        x = state_matrix[:, 0]
        y = state_matrix[:, 1]
        z = state_matrix[:, 2]

        xdot = self.sigma * (y - x)
        ydot = x * (self.rho - z) - y
        zdot = x * y - self.beta * z

        # Assemble again the state matrix
        dstate_matrix = np.stack([xdot, ydot, zdot], axis=1)

        # Coupling term
        dstate_matrix = dstate_matrix + (self.H @ state_matrix)

        return self._to_array(dstate_matrix)

    def simulate(self, length=50, dt=0.015, init_state=np.array(([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))):
        input_state = self._to_array(init_state)
        t = np.linspace(0, length, int(length/dt))
        trajectories = odeint(self.system_fn, input_state, t)
        trajectories = trajectories.reshape(-1, self.num_nodes, self.ndim)

        return trajectories



class LorenzDataset(SyntheticDataset):
    """
    Lorenz attractor dataset
    """
    def __init__(self, name, save_dir=None, reprocess=False, 
                 num_samples=10, num_nodes=3, dt=0.1, duration=10, k=0.01,                 
                 space_coupling=1, time_coupling=1, has_labels=False, 
                 save_info=True):
        self._name = f"{name}"
        self.has_labels = has_labels
        self.save_info = save_info
        self._save_path = os.path.join(save_dir, self.name)   
        if not os.path.isdir(self._save_path):
            reprocess = True  # For sure, since we don't have the data
        else:
            if reprocess:
                shutil.rmtree(self._save_path)
                os.makedirs(self._save_path, exist_ok=True)
        # This dataset has no labels and no additional information
        self.has_labels = has_labels
        self.save_info = save_info

        # ODE system
        self.system_coupling = k
        # self.system_fn = Lorenz(sigma=10, rho=28, beta=8/3)
        self.system_fn = CoupledLorenz(num_nodes=num_nodes, k=self.system_coupling, sigma=10, rho=28, beta=8/3)

        # Number of samples
        self.num_samples = num_samples
        self.dt = dt
        self.duration = duration

        # Number of nodes / or 'components' of the system
        self.num_nodes = num_nodes

        # Joint parameters
        self.space_coupling = space_coupling  # Coupling strength in space
        self.time_coupling = time_coupling  # Coupling strength in time

        # Name of the features [states]
        self.features = ['$x$', '$y$', '$z$']

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
                      save_folder=self._save_path,
                      predicted_length=self.duration
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
    # By default
    paper_folder = "/media/jaume/DATA/Data/Multiplex_Synthetic_FINAL"
    # paper_folder = "/usr/data/Multiplex_Synthetic_FINAL"
    # study_name = "Multiplex_Lorenz"
    # study_name = "Multiplex_Lorenz_ADAM_FINAL_MAE"
    # study_name = "Multiplex_Lorenz_ADAM_END"
    # study_name = "Multiplex_Lorenz_DIMENSIONS"
    study_name = "Multiplex_Lorenz_DIMENSIONS_NEW_LOSS"

    model_to_load_test = os.path.join(paper_folder, 'LorenzAttractor_Graphs', 'duration-2.5_dt-0.05_k-0.01_nodes-3', 'Test_Run', 'model.pt')
    model_to_load_test = None
    
    parser = argparse.ArgumentParser(description='Lorenz Attractor')

    # Folder
    parser.add_argument('--save_folder', type=str, default=f"{paper_folder}", help='Folder to save the data')
    parser.add_argument('--name', type=str, default='LorenzAttractor', help='Name of the dataset')
    parser.add_argument('--reprocess', type=str2bool, default=False, help='Reprocess the data')
    parser.add_argument('--reload_model', type=str2bool, default=True, help='Reload the model - if it exists')
    parser.add_argument('--model_to_load', type=str, default=model_to_load_test, help='Loads another model.')
    parser.add_argument('--fine_tune', type=str2bool, default=False, help='Checked in case that we load another model. If so, fine tune the model by the number of epochs. Otherwise zero-shot.')
    parser.add_argument('--experiment_name', type=str, required=False, default=f'{study_name}', help='Name of the experiment.')
    parser.add_argument('--run_best', type=str2bool, default=True, help='Run the best model')
    parser.add_argument('--is_optuna', type=str2bool, default=False, help='Run the best model')

    # Simulation
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--num_nodes', type=int, default=3, help='Number of nodes')
    parser.add_argument('--k', type=float, default=0.01, help='Coupling strength [0-1]')
    parser.add_argument('--space_coupling', type=float, default=1, help='Space coupling')
    parser.add_argument('--time_coupling', type=float, default=1, help='Time coupling')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step')
    parser.add_argument('--duration', type=float, default=2.5, help='Duration of the simulation')

    # Model parameters
    parser.add_argument('--normalization', type=str, default='ZNorm', help='Normalization method')
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
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials for the hyper-parameter tuning')
    parser.add_argument('--num_jobs', type=int, default=0, help='Number of jobs in parallel')
    
    parser.add_argument('--gamma_rec', type=float, default=1., help='Weight for the regression loss')
    parser.add_argument('--gamma_class', type=float, default=0., help='Weight for the classification loss')
    parser.add_argument('--gamma_bc', type=float, default=0., help='Weight for the boundary condition in the latent space')
    parser.add_argument('--gamma_lat', type=float, default=0.05, help='L2 weight for the latent space')
    parser.add_argument('--gamma_graph', type=float, default=0., help='Weight for the graph regularization')

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
    dataset_name = f'duration-{duration}_dt-{dt}_k-{k}_nodes-{num_nodes}'

    # Save folder
    save_folder = os.path.join(args.save_folder, "LorenzAttractor_Graphs", f"{dataset_name}")
    os.makedirs(save_folder, exist_ok=True)

    # Store .json with the configuration
    json_filename = os.path.join(save_folder, 'config.json')
    with open(json_filename, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    normalization = args.normalization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    # lorenz = Lorenz(sigma=10, rho=28, beta=8/3)
    # generate_data(lorenz, num_samples=500, length=10, dt=0.1,num_nodes=2, space_coupling=0.1, time_coupling=0.1, save_folder=save_folder)    
    is_optuna = args.is_optuna
    # if is_optuna:
    #     num_samples = 400  # Just for the hyper-parameter tuning, otherwise takes too long.
    # else:
    num_samples = args.num_samples

    # Create the dataset
    dataset = LorenzDataset(name=args.name, 
                            save_dir=save_folder, 
                            reprocess=args.reprocess, 
                            k=k, 
                            num_nodes=num_nodes,
                            space_coupling=args.space_coupling, 
                            time_coupling=args.time_coupling, 
                            dt=dt, 
                            duration=duration, 
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
    # dt_step_size = 0.04
    
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

        results_hp_folder = os.path.join(args.save_folder, "LorenzAttractor_Graphs", "results")
        df_params_path = os.path.join(results_hp_folder, f'{study_name}_trials.csv')
        df_params = pd.read_csv(df_params_path)
        df_params.dropna(how='any', inplace=True)
        df_params = df_params.sort_values(by='value', ascending=False)
        # best_params = df_params.iloc[0].to_dict()
        # # best_params = df_params.iloc[0:5][params_names].mean().to_dict()
        # best_params = {key.replace('params_', ''): value for key, value in best_params.items() if key.startswith('params_')}
        # df_params['params_decode_just_latent'] = True  # HARDCODED
        best_params = get_best_params(df_params.iloc[0:5], use_median=True)

        print(f"Best parameters: {best_params}")
        model_params = objective_optuna.default_params.copy()
        model_params.update(best_params)

        # Store the parameters
        json_filename = os.path.join(save_folder, f'objective_params_{study_name}.json')
        with open(json_filename, 'w') as f:
            json.dump(model_params, f, indent=4)

        # Model
        # model_to_load = args.model_to_load
        # fine_tune_model = args.fine_tune
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
    
    if sq_database or run_best or not is_optuna:
        steps_to_predict = int(duration/dt)
        # steps_to_predict = int(duration / dt_step_size)
        print(steps_to_predict)
        # time_to_predict = np.arange(0, steps_to_predict, 1)  # Predict 100 steps more
        time_to_predict = torch.arange(0, steps_to_predict, 1)
        pred_trajectory, pred_latent, tgt_trajectory = objective_optuna.predict_from_latent(model, objective_optuna.dataset, time_to_predict, model_params, device=device)
        # The shape of the results is [num_samples, num_features, num_nodes, num_time_steps]

        # Save as metric the error in the predicted trajectory
        # total_mse = (tgt_trajectory - pred_trajectory.mean).square().mean(dim=0).sum()
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
        wrapped_table = wrap_latex_table(latex_table, caption="Errors per feature in the test set", label="tab:feature_errors_lorenz")
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