import dgl
import torch
import numpy as np
import networkx as nx


def generate_graph(num_nodes, graph_type, graph_params):
    # Define adjacency matrix for spatial coupling
    if graph_type == 'fully_connected':
        A = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    elif graph_type == 'small_world':
        k = graph_params.get('k', num_nodes//2)  # Each node connects to 'k' nearest neighbors
        p = graph_params.get('p', 0.5)  # Rewiring probability
        G = nx.watts_strogatz_graph(num_nodes, k, p)
        A = nx.adjacency_matrix(G).toarray()
    elif graph_type == 'barabasi':        
        m = graph_params.get('m', num_nodes // 2)  # Number of edges to attach from a new node to existing nodes
        G = nx.barabasi_albert_graph(num_nodes, m)
        A = nx.adjacency_matrix(G).toarray()
    elif graph_type == 'erdos':
        p = graph_params.get('p', 0.5)  # Probability of edge creation
        G = nx.erdos_renyi_graph(num_nodes, p)
        A = nx.adjacency_matrix(G).toarray()
    elif graph_type == 'identity':
        A = np.eye(num_nodes)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}. Choose 'fully_connected', 'identity', 'small_world', 'barabasi', or 'erdos'.")

    return A


def sparse_diag(input):
    N = input.shape[0]
    arr = torch.arange(N, device=input.device)
    indices = torch.stack([arr, arr])
    return torch.sparse_coo_tensor(indices, input, (N, N))


def graph_to_normalized_adjacency(graph):
    # --- DON"T USE IT BECAUSE IT FAILS WHEN THE GRAPH HAS ONLY SELF-LOOPS ---
    graph = graph.remove_self_loop()  # To be sure of not having duplicated self-loops
    normalized_graph = dgl.GCNNorm()(graph.add_self_loop())
    adj = normalized_graph.adj(ctx=graph.device)
    new_adj = torch.sparse_coo_tensor(
        adj.coalesce().indices(), normalized_graph.edata['w'], tuple(adj.shape)
    )
    return new_adj


def weighted_adjacency_to_graph(adj, edata_name='w'):
    adj = adj.coalesce()
    indices = adj.indices()
    graph = dgl.graph((indices[0], indices[1]))
    # graph.edata[f"{edata_name}"] = adj.values()
    graph.edata[f"{edata_name}"] = adj.val

    return graph


def normalize_graph(graph, 
                    ord='sym', 
                    etype=None,
                    add_self_loop=False,
                    copy_ndata=True,
                    copy_edata=True,
                    norm_edata=None):
    if add_self_loop:
        graph = graph.remove_self_loop()  # To be sure of not having duplicated self-loops
        graph = graph.add_self_loop()
    
    if norm_edata is None:
        # Use the adjacency matrix
        # adj = graph.adj(ctx=graph.device, etype=etype).coalesce()
        adj = graph.adj(etype=etype).coalesce()
        # adj = torch.sparse_coo_tensor(indices, input, (N, N))
    else:
        # Use the edge data to normalise
        #TODO: Extend to use multi-dimensional edges
        indices = torch.stack(graph.edges(etype=etype))        
        adj = torch.sparse_coo_tensor(
            indices,
            graph.edata[norm_edata], 
            (graph.number_of_nodes(), graph.number_of_nodes())
        )
    
    if ord == 'row':
        norm = adj.sum(dim=1)
        # norm = torch.sparse.sum(adj, dim=1).to_dense()
        norm[norm<=0] = 1
        inv_D = dgl.sparse.diag(1/norm)
        # inv_D = sparse_diag(1 / norm)
        new_adj = inv_D @ adj
    elif ord == 'col':
        # norm = torch.sparse.sum(adj, dim=0).to_dense()
        norm = adj.sum(dim=0)
        norm[norm<=0] = 1
        # inv_D = sparse_diag(1 / norm)
        inv_D = dgl.sparse.diag(1/norm)
        new_adj = adj @ inv_D
    elif ord == 'sym':
        norm = adj.sum(dim=1)
        # norm = torch.sparse.sum(adj, dim=1).to_dense()
        norm[norm<=0] = 1
        # inv_D = sparse_diag(norm ** (-0.5))
        inv_D = dgl.sparse.diag(norm ** (-0.5))
        new_adj = inv_D @ adj @ inv_D

    new_graph = weighted_adjacency_to_graph(new_adj, edata_name='w')

    return new_graph
