import os
import numpy as np
from typing import Union

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
utils = importr('utils')
utils.chooseCRANmirror(ind=1) 
utils.install_packages('FOCI', quiet=True)


__all__ = ["est_struct_shifts"]


def _estimate_parents_foci(X: np.ndarray, 
                          node_set: np.ndarray, 
                          order: np.ndarray) -> np.ndarray:
    """
    Uses FOCI [1] to estimate local parents of ``node_set`` given a topological order in ``order``.

    References
    ----------
    1. Azadkia and Chatterjee (2019),"A simple measure of conditional dependence" <arXiv:1910.12327>.

    Parameters
    ----------
    X : np.ndarray
        Dataset [n, d]
    node_set : np.ndarray
        Set of nodes of interest        
    order : np.ndarray
        Topological ordering

    Returns
    -------
    np.ndarray
        Adjacency matrix [d,d] with estimation of local parents
    """
    rpy2.robjects.numpy2ri.activate()
    nr, nc = X.shape
    X = ro.r.matrix(X, nr, nc)

    # R indexes start from 1
    node_set = ro.r.matrix(node_set + 1, 1, len(node_set))
    order = ro.r.matrix(order + 1, 1, len(order))

    path = os.path.dirname(os.path.abspath(__file__))
    ro.r.source(os.path.join(path, "my_foci.R"))
    local_pa = ro.r.find_par(X, node_set, order)
    return local_pa


def est_struct_shifts(X: np.ndarray, 
                      Y: np.ndarray, 
                      shifted_nodes: Union[list, np.ndarray], 
                      order: Union[list, np.ndarray],
                      method: str = "foci",
                      ) -> np.ndarray:
    """
    Estimates structural changes from data for a subset of variables (shifted nodes),
    given an order of the variables.

    Parameters
    ----------
    X : np.ndarray
        Dataset X
    Y : np.ndarray
        Dataset Y
    shifted_nodes : Union[list, np.ndarray]
        Set of variables of interest
    order : Union[list, np.ndarray]
        Valid topological ordering of the variables
    method : str, optional
        method to use for estimation of local parents. One of ``["foci"]``.

    Returns
    -------
    np.ndarray
        Estimation of structural changes for the set of shifted nodes
    """
    shifted_nodes, order = np.array(shifted_nodes), np.array(order)
    if method == 'foci':
        x_adj = _estimate_parents_foci(X, shifted_nodes, order)
        y_adj = _estimate_parents_foci(Y, shifted_nodes, order)
    else:
        raise NotImplementedError("method not implemented. Options are: ['foci']")
    ddag = np.array(np.abs(x_adj - y_adj), dtype=int)
    return ddag


def test():
    from .utils import DataGenerator, ddag_metrics, set_seed
    set_seed(24)
    d, s0 = 20, 60
    generator = DataGenerator(d, s0, "ER")
    X, Y = generator.sample(10000, num_shifted_nodes=4, change_struct=True)
    shifted_nodes, order = generator.shifted_nodes, np.arange(d)
    true_ddag = np.abs(generator.adj_X - generator.adj_Y, dtype=int)
    est_ddag = est_struct_shifts(X, Y, shifted_nodes, order)
    print(ddag_metrics(true_ddag, est_ddag))


if __name__ == "__main__":
    test()

