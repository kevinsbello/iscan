import random
import torch
from torch.distributions import MultivariateNormal, Normal, Laplace, Gumbel
import os
from sklearn.metrics import precision_score,recall_score,f1_score
import numpy as np
import igraph as ig
import GPy

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def nodes_metrics(true_shifted_nodes, predict_shifted_node,d):
    # true_shifted_nodes is the list for true shifted nodes
    # predict_shifted_node is the list for predicted shifted nodes
    # d is the number of nodes
    y_actual = np.zeros(d)
    y_pred = np.zeros(d)
    y_actual[true_shifted_nodes] = 1
    y_pred[predict_shifted_node] = 1

    prec = precision_score(y_actual,y_pred, zero_division=0)
    recall = recall_score(y_actual, y_pred, zero_division=0)
    f1 = f1_score(y_actual, y_pred, zero_division=0)
    return [prec,recall,f1]

def ddag_metrics(adj_true,adj_pred):
    prec = precision_score(adj_true.flatten(),adj_pred.flatten(),zero_division=0)
    recall = recall_score(adj_true.flatten(),adj_pred.flatten(),zero_division=0)
    f1 = f1_score(adj_true.flatten(),adj_pred.flatten(), zero_division=0)
    return [prec,recall,f1]

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_acyclic_orientation(B_und):
        return np.triu(B_und, k=1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
        B_und = _graph_to_adjmat(G)
        B = _random_acyclic_orientation(B_und)
    else:
        raise ValueError('unknown graph type')
    assert ig.Graph.Adjacency(B.tolist()).is_dag()
    return B


############################################################
# Below: Generate data for shifted nodes and shifted edges #
############################################################

class DataGenerator:
    def __init__(self,n,d,s0,graph_type,noise_std,noise_type):
        self.n,self.noise_std, self.noise_type = n,noise_std,noise_type
        self.d, self.s0, self.graph_type = d,s0,graph_type

        self.adjacency = simulate_dag(d, s0, graph_type)

        # init for adjacency deleted for edges part
        self.adjacency_del = self.adjacency.copy()
        noise_std = noise_std * torch.ones(d)

        if noise_type == "Gaussian":
            noise_dist = Normal(0, noise_std)
        elif noise_type == "Laplace":
            noise_dist = Laplace(0, noise_std)
        elif noise_type == "Gumbel":
            noise_dist = Gumbel(0, noise_std)
        else:
            raise ValueError('unknown noise type')

        self.noise_X = noise_dist.sample((n,))  # n x d noise matrix
        self.noise_Y = noise_dist.sample((n,))  # n x d noise matrix
        self.X = torch.zeros(n, d)
        self.Y = torch.zeros(n, d)

    def sample_GP(self,X,lengthscale=1/2):
        ker = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=1)
        C = ker.K(X, X)
        ## the non linear part is from a gaussian distribution.
        X_sample = np.random.multivariate_normal(np.zeros(len(X)), C)
        return X_sample

    def sample_nodes(self,adjacency,shift_index,GP=False):
        if GP:
            for i in range(self.d):
                self.X[:, i] = self.noise_X[:, i]
                self.Y[:, i] = self.noise_Y[:, i]

                par = np.nonzero(adjacency[:, i])[0]
                if len(par) == 0:
                    pass
                elif i in shift_index:
                    X_par = self.X[:, par]
                    Y_par = self.Y[:, par]

                    self.X[:, i] += torch.tensor(self.sample_GP(np.array(X_par), 1/2))
                    self.Y[:, i] += torch.tensor(self.sample_GP(np.array(Y_par), 1/2)) * 2
                else:
                    X_par = self.X[:, par]
                    Y_par = self.Y[:, par]

                    combine = torch.concat((X_par, Y_par))
                    combine_gp = torch.tensor(self.sample_GP(np.array(combine), 1/2))
                    # First half is for X
                    # Second half is for Y
                    self.X[:, i] += combine_gp[:self.n]
                    self.Y[:, i] += combine_gp[self.n:]
        else:
            for i in range(self.d):
                self.X[:, i] = self.noise_X[:, i]
                self.Y[:, i] = self.noise_Y[:, i]
                for j in np.nonzero(adjacency[:, i])[0]:
                    self.X[:, i] += torch.sin(torch.square(self.X[:, j]))
                    if i in shift_index:
                        self.Y[:, i] += 4 * torch.cos(2 * torch.square(self.Y[:, j]) - 3 * self.Y[:, j])
                    else:
                        self.Y[:, i] += torch.sin(torch.square(self.Y[:, j]))

    def generate_nodes_data(self,n_shift_node,GP = False):
        root = np.where(self.adjacency.sum(axis=0) == 0)[0]
        non_root = np.setdiff1d(list(range(self.d)), root)
        # uniformly choose shift node
        self.shift_index = np.random.choice(non_root, min(n_shift_node, len(non_root)), replace=False)
        self.sample_nodes(self.adjacency, self.shift_index, GP)
        return self.X, self.Y, self.shift_index, self.adjacency

    def random_delete_edge(self, proportion):
        """
        Generate new adjacency by randomly deleting edges based on a given adjacency
        :param proportion: proportion of edges to be deleted
        :return: deleted edges adjacency matrix
        """

        root = np.where(np.sum(self.adjacency, axis=0) == 0)[0]
        non_root = [x for x in range(self.d) if x not in root]
        hub_in_degree_node = [i for i in non_root if np.sum(self.adjacency[:, i]) >= 2]

        num_change = int(self.d * proportion)
        num_change = min(num_change, len(hub_in_degree_node))
        change_nodes = np.random.choice(hub_in_degree_node, size=num_change, replace=False)

        for k in change_nodes:
            par = np.nonzero(self.adjacency[:, k])[0]
            # delete 3 edges
            num_del = 3
            del_par = np.random.choice(par, size=min(num_del, len(par)), replace=False)
            self.adjacency_del[del_par, k] = 0
        return self.adjacency_del, change_nodes

    def sample_edges(self,noise_matrix,_adjacency,ddag):
        for i in range(self.d):
            for j in np.nonzero(_adjacency[:, i])[0]:
                if ddag[j, i] == 1:
                    # only change function with deleted edges
                    noise_matrix[:, i] += 4 * torch.cos(2 * torch.square(noise_matrix[:, j]) - 3 * noise_matrix[:, j])
                else:
                    noise_matrix[:, i] += torch.sin(torch.square(noise_matrix[:, j]))
        return None

    def generate_pair_adjacency(self,delete_prop):
        x_adjacency = self.adjacency
        y_adjacency, change_nodes = self.random_delete_edge(delete_prop)
        ddag = (x_adjacency - y_adjacency != 0).astype(int)
        return (x_adjacency, y_adjacency, ddag, change_nodes)

    def generate_edges_data(self,delete_prop):
        x_adjacency, y_adjacency, ddag, changed_nodes = self.generate_pair_adjacency(delete_prop)
        self.X = self.noise_X.clone()
        self.Y = self.noise_Y.clone()
        self.sample_edges(self.X,x_adjacency, ddag)
        self.sample_edges(self.Y,y_adjacency, ddag)
        return self.X, self.Y, changed_nodes,ddag
