import random, os
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import igraph as ig
import GPy
from typing import Union, List, Tuple


__all__ = ["set_seed", "node_metrics", "ddag_metrics", "DataGenerator"]


def set_seed(seed: int = 42) -> None:
    """
    Sets random seed of ``random`` and ``numpy`` to specified value
    for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Value for RNG, by default 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def node_metrics(true_shifted_nodes: list, pred_shifted_nodes: list, d: int) -> dict:
    """
    Computes precision, recall, and f1 scores for the predicted shifted nodes.
    
    Parameters
    ----------
    true_shifted_nodes : list
        List of true shifted nodes
    pred_shifted_nodes : list
        List of predicted shifted nodes
    d : int
        Total number of nodes/variables

    Returns
    -------
    dict
        precision, recall, and F1 scores
    """
    y_actual = np.zeros(d)
    y_pred = np.zeros(d)
    y_actual[list(true_shifted_nodes)] = 1
    y_pred[list(pred_shifted_nodes)] = 1

    prec = precision_score(y_actual,y_pred, zero_division=0)
    recall = recall_score(y_actual, y_pred, zero_division=0)
    f1 = f1_score(y_actual, y_pred, zero_division=0)
    scores = {"prec": prec, "recall": recall, "f1": f1}
    return scores


def ddag_metrics(adj_true: np.ndarray, adj_pred: np.ndarray) -> dict:
    """
    Computes precision, recall, and f1 scores for the predicted structural changes/shifts (difference DAG).

    Parameters
    ----------
    adj_true : np.ndarray
        Adjancecy matrix of the true difference DAG (graph of structural differences)
    adj_pred : np.ndarray
        Predicted adjancecy matrix of the difference DAG (graph of structural differences)

    Returns
    -------
    dict
        Precision, recall, and F1 scores
    """
    prec = precision_score(adj_true.flatten(),adj_pred.flatten(),zero_division=0)
    recall = recall_score(adj_true.flatten(),adj_pred.flatten(),zero_division=0)
    f1 = f1_score(adj_true.flatten(),adj_pred.flatten(), zero_division=0)
    scores = {"prec": prec, "recall": recall, "f1": f1}
    return scores


class DataGenerator:
    """
    Generate synthetic data to evaluate shifted nodes and shifted edges
    """
    def __init__(self, d: int, s0: int, graph_type: str, 
                 noise_std: Union[float, np.ndarray, List[float]] = .5, 
                 noise_type: str = "Gaussian",
                ):
        """
        Defines the class of graphs and data to be generated.

        Parameters
        ----------
        d : int
            Number of variables
        s0 : int
            Expected number of edges in the random graphs
        graph_type : str
            One of ``["ER", "SF"]``. ``ER`` and ``SF`` refer to Erdos-Renyi and Scale-free graphs, respectively.
        noise_std : Union[float, np.ndarray, List[float]], optional
            Sets the scale (variance) of the noise distributions. Should be either a scalar, or a list (array) of dim d. By default 1.
        noise_type : str, optional
            One of ``["Gaussian", "Laplace", "Gumbel"]``, by default "Gaussian".
        """
        self.d, self.s0, self.graph_type = d, s0, graph_type
        self.noise_type = noise_type
        self.noise_std = noise_std
        if np.isscalar(noise_std):
            self.noise_std = np.ones(d) * noise_std
        else:
            if len(noise_std) != d:
                raise ValueError('noise_std must be a scalar or have length d')
        if noise_type == "Gaussian":
            self.noise_dist = np.random.normal
        elif noise_type == "Laplace":
            self.noise_dist = np.random.laplace
        elif noise_type == "Gumbel":
            self.noise_dist = np.random.gumbel
        else:
            raise ValueError('Unknown noise type. Should be one of ["Gaussian", "Laplace", "Gumbel"]')
    
    def _simulate_dag(self, d: int, s0: int, graph_type: str) -> np.ndarray:
        """
        Simulate random DAG with some expected number of edges.

        Parameters
        ----------
        d : int
            num of nodes
        s0 : int
            expected num of edges
        graph_type : str
            ER, SF

        Returns
        -------
        np.ndarray
            :math:`(d, d)` binary adj matrix of a sampled DAG
        """
        def _random_acyclic_orientation(B_und):
            return np.triu(B_und, k=1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == 'ER': # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == 'SF': # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
            B_und = _graph_to_adjmat(G)
            B = _random_acyclic_orientation(B_und)
        else:
            raise ValueError('unknown graph type')
        
        assert ig.Graph.Adjacency(B.tolist()).is_dag()
        
        return B
    
    def _sample_GP(self, X: np.ndarray, lengthscale: float = .5):
        ker = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=1)
        C = ker.K(X, X)
        X_sample = np.random.multivariate_normal(np.zeros(len(X)), C)
        return X_sample

    def _choose_shifted_nodes(self, adj: np.ndarray, num_shifted_nodes: int) -> np.ndarray:
        """
        Randomly choose shifted nodes

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix
        num_shifted_nodes : int
            Number of desired shifted nodes

        Returns
        -------
        np.ndarray
            Shifted nodes
        """
        roots = np.where(adj.sum(axis=0) == 0)[0]
        non_roots = np.setdiff1d(list(range(self.d)), roots)
        hubs = [i for i in non_roots if np.sum(adj[:, i]) >= 2]
        # uniformly choose shift node
        shifted_nodes = np.random.choice(hubs, min(num_shifted_nodes, len(hubs)), replace=False)
        return shifted_nodes
    
    def _delete_edges(self, adj: np.ndarray, shifted_nodes: np.ndarray) -> np.ndarray:
        """
        Generates a new adjacency matrix where some incoming edges (parents) are randomly deleted for the shifted nodes.

        Parameters
        ----------
        adj : np.ndarray
            Base adjacency matrix
        shifted_nodes : np.ndarray
            Vector of shifted node indices

        Returns
        -------
        np.ndarray
            [d, d] adjacency matrix with removed edges
        """
        _adj = adj.copy()
        for k in shifted_nodes:
            par = np.nonzero(adj[:, k])[0]
            num_del = 3 # randomly delete 3 edges if possible
            del_par = np.random.choice(par, size=min(num_del, len(par)), replace=False)
            _adj[del_par, k] = 0
        return _adj

    def _sample_from_same_structs(self, adj: np.ndarray, 
                                  shifted_nodes: np.ndarray, 
                                  use_gp: bool = False
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample data where both DAGs have the same structure

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix
        shifted_nodes : np.ndarray
            Set of shited nodes
        use_gp : bool, optional
            Whether to sample functions from Gaussian Processes, by default ``False``

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Datasets X and Y
        """
        dataX, dataY = np.zeros((self.n, self.d)), np.zeros((self.n, self.d))
        for i in range(self.d):
            dataX[:, i] = self.noise_dist(scale=self.noise_std[i], size=self.n) 
            dataY[:, i] = self.noise_dist(scale=self.noise_std[i], size=self.n) 
        if use_gp:
            for i in range(self.d):
                pa = np.nonzero(adj[:, i])[0]
                if len(pa) == 0:
                    continue
                X_pa = dataX[:, pa]
                Y_pa = dataY[:, pa]
                if i in shifted_nodes:
                    dataX[:, i] += self._sample_GP(X_pa, .5)
                    dataY[:, i] += self._sample_GP(Y_pa, .5) * 2
                else:
                    combine_gp = self._sample_GP(np.concatenate((X_pa, Y_pa)), .5) # sample from same GP
                    dataX[:, i] += combine_gp[:self.n]
                    dataY[:, i] += combine_gp[self.n:]
        else:
            for i in range(self.d):
                for j in np.nonzero(adj[:, i])[0]:
                    dataX[:, i] += np.sin(dataX[:, j] ** 2)
                    dataY[:, i] += np.sin(dataY[:, j] ** 2) if i not in shifted_nodes \
                                    else 4 * np.cos(2 * dataY[:, j]**2 - 3 * dataY[:, j])
        return dataX, dataY 
    
    def _sample_from_diff_structs(self, adj_x: np.ndarray, adj_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample data where both DAGs have the different structures

        Parameters
        ----------
        adj_x : np.ndarray
            Adjacency matrix of DAG X
        adj_y : np.ndarray
            Adjacency matrix of DAG Y

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Datasets X and Y
        """
        dataX, dataY = np.zeros((self.n, self.d)), np.zeros((self.n, self.d))
        for i in range(self.d):
            dataX[:, i] = self.noise_dist(scale=self.noise_std[i], size=self.n) 
            dataY[:, i] = self.noise_dist(scale=self.noise_std[i], size=self.n) 
        for i in range(self.d):
            pa_x, pa_y = np.nonzero(adj_x[:, i])[0], np.nonzero(adj_y[:, i])[0]
            for j in range(self.d):
                if j in pa_x and j in pa_y:
                    dataX[:, i] += np.sin(dataX[:, j] ** 2)
                    dataY[:, i] += np.sin(dataY[:, j] ** 2)
                elif j in pa_x:
                    dataX[:, i] += 4 * np.cos(2 * dataX[:, j]**2 - 3 * dataX[:, j])
                elif j in pa_y:
                    dataY[:, i] += 4 * np.cos(2 * dataY[:, j]**2 - 3 * dataY[:, j])
        return dataX, dataY

    def sample(self, n: int, 
               num_shifted_nodes: int, 
               change_struct: bool = False, 
               use_gp: bool = False,
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples two datasets from randomly generated DAGs

        Parameters
        ----------
        n : int
            Number of samples
        num_shifted_nodes : int
            Desired number of shifted nodes. Actual number can be lower.
        change_struct : bool, optional
            Whether or not to sample from DAGs with different structures, by default ``False``.
        use_gp : bool, optional
            Whether or not to sample from Gaussian Processes, by default ``True``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two datasets with shape [n, d]

        Raises
        ------
        NotImplementedError
            The case of sampling from GPs when the structures are differents is not implemented.
        """
        if change_struct and use_gp:
            raise NotImplementedError("One of use_gp and change_struct needs to be False")
        self.n = n
        self.adj_X = self._simulate_dag(self.d, self.s0, self.graph_type)
        self.shifted_nodes = self._choose_shifted_nodes(self.adj_X, num_shifted_nodes)
        self.adj_Y = self.adj_X.copy() if not change_struct else self._delete_edges(self.adj_X, self.shifted_nodes)
        if change_struct:
            dataX, dataY = self._sample_from_diff_structs(self.adj_X, self.adj_Y)
        else:
            dataX, dataY = self._sample_from_same_structs(self.adj_X, self.shifted_nodes, use_gp)
        return dataX, dataY

