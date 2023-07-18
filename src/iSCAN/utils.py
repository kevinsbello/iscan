import random
import torch
import os
from sklearn.metrics import precision_score,recall_score,f1_score
import numpy as np
import igraph as ig

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

def nodes_metrics(y_actual, y_pred):
    # y_actual is the list for true shifted nodes
    # y_pred is the list for predicted shifted nodes
    prec = precision_score(y_actual,y_pred, zero_division=0)
    recall = recall_score(y_actual, y_pred, zero_division=0)
    f1 = f1_score(y_actual, y_pred, zero_division=0)
    return [prec,recall,f1]

def ddag_metrics(adj_true,adj_pred):
    shd = np.sum(adj_true!=adj_pred)
    prec = precision_score(adj_true.flatten(),adj_pred.flatten(),zero_division=0)
    recall = recall_score(adj_true.flatten(),adj_pred.flatten(),zero_division=0)
    return [shd,prec,recall]

def dag_simulator(d, s0, graph_type, triu=True):
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