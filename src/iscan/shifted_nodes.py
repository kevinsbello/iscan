import pandas as pd
from kneed import KneeLocator
import numpy as np
import torch
from .score_estimator import stein_hess
from typing import Union, Tuple


__all__= ["est_node_shifts"]


def _find_elbow(diff_dict: dict, hard_thres: float = 30, online: bool = True) -> np.ndarray:
    r"""
    Return selected shifted nodes by finding elbow point on sorted variance

    Parameters
    ----------
    diff_dict : dict
        A dictionary where ``key`` is the index of variables/nodes, and ``value`` is its variance ratio.
    hard_thres : float, optional
        | Variance ratios larger than hard_thres will be directly regarded as shifted. 
        | Selected nodes in this step will not participate in elbow method, by default 30.
    online : bool, optional
        If ``True``, the heuristic will find a more aggressive elbow point, by default ``True``.

    Returns
    -------
    np.ndarray
        A dict with selected nodes and corresponding variance
    """
    diff = pd.DataFrame()
    diff.index = diff_dict.keys()
    diff["ratio"] = [x for x in diff_dict.values()]
    shift_node_part1 = diff[diff["ratio"] >= hard_thres].index
    undecide_diff = diff[diff["ratio"] < hard_thres]
    kn = KneeLocator(range(undecide_diff.shape[0]), undecide_diff["ratio"].values,
                     curve='convex', direction='decreasing',online=online,interp_method="interp1d")
    shift_node_part2 = undecide_diff.index[:kn.knee]
    shift_node = np.concatenate((shift_node_part1,shift_node_part2))
    return shift_node


def _get_min_rank_sum(HX: torch.Tensor, HY: torch.Tensor) -> int:
    r"""
    | Find which node has the mininum rank sum in datasets X and dataset Y.
    | This is helpful to select a common leaf across the datasets.
    
    Parameters
    ----------
    HX : torch.Tensor
        Hessian's diagonal estimation for dataset X.
    HY : torch.Tensor
        Hessian's diagonal estimation for dataset Y.

    Returns
    -------
    int
        Node index that has the smallest rank sum.
    """
    order_X = torch.argsort(HX.var(axis=0))
    rank_X = torch.argsort(order_X)

    order_Y = torch.argsort(HY.var(axis=0))
    rank_Y = torch.argsort(order_Y)
    l = int((rank_X + rank_Y).argmin())
    return l


def est_node_shifts(
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor], 
        eta_G: float = 0.001, 
        eta_H: float = 0.001,
        normalize_var: bool = False, 
        shifted_node_thres: float = 2.,
        elbow: bool = False,
        elbow_thres: float = 30.,
        elbow_online: bool = True,
        use_both_rank: bool = True,
        verbose: bool = False,
    ) -> Tuple[list, list, dict]:
    r"""
    | Implementation of the iSCAN method of Chen et al. (2023).
    | Returns an estimated topological ordering, and estimated shifted nodes
    
    References
    ----------
    - T. Chen, K. Bello, B. Aragam, P. Ravikumar. (2023). 
    `iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models. <https://arxiv.org/abs/2306.17361>`_.
    
    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor]
        Dataset with shape :math:`(n,d)`, where :math:`n` is the number of samples, 
        and :math:`d` is the number of variables/nodes.
    Y : Union[np.ndarray, torch.Tensor]
        Dataset with shape :math:`(n,d)`, where :math:`n` is the number of samples, 
        and :math:`d` is the number of variables/nodes.
    eta_G : float, optional
        hyperparameter for the score's Jacobian estimation, by default 0.001.
    eta_H : float, optional
        hyperparameter for the score's Jacobian estimation, by default 0.001.
    normalize_var : bool, optional
        If ``True``, the Hessian's diagonal is normalized by the expected value, by default ``False``.
    shifted_node_thres : float, optional
        Threshold to decide whether or not a variable has a distribution shift, by default 2.
    elbow : bool, optional
        If ``True``, iscan uses the elbow heuristic to determine shifted nodes. By default ``True``.
    elbow_thres : float, optional
        If using the elbow method, ``elbow_thres`` is the ``hard_thres`` in 
        :py:func:`~iscan-dag.shifted_nodes.find_elbow` function, by default 30.
    elbow_online : bool, optional
        If using the elbow method, ``elbow_online`` is ``online`` in 
        :py:func:`~iscan-dag.shifted_nodes.find_elbow` function, by default ``True``.
    use_both_rank : bool, optional
        estimate topo order by X's and Y's rank sum. If False, only use X for topo order, by default ``False``.
    verbose : bool, optional
        If ``True``, prints to stdout the variances of the Hessian entries for the running leafs.
        By default ``False``.
    Returns
    -------
    Tuple[list, list, dict]
        estimated shifted nodes, topological order, and dict of variance ratios for all nodes.
    """
    torch.set_default_dtype(torch.double)
    vprint = print if verbose else lambda *a, **k: None

    if type(X) is np.ndarray:
        X = torch.from_numpy(X)
    if type(Y) is np.ndarray:
        Y = torch.from_numpy(Y)
    
    X, Y = X.double(), Y.double()
    n, d = X.shape
    order = [] # estimates a valid topological sort
    active_nodes = list(range(d))
    shifted_nodes = [] # list of estimated shifted nodes
    dict_stats = dict() # dictionary of variance ratios for all nodes

    for _ in range(d - 1):
        A = torch.concat((X, Y))
        HX = stein_hess(X, eta_G, eta_H)
        HY = stein_hess(Y, eta_G, eta_H)
        if normalize_var:
            HX = HX / HX.mean(axis=0)
            HY = HY / HY.mean(axis=0)
        
        # estimate common leaf
        l = _get_min_rank_sum(HX,HY) if use_both_rank else int(HX.var(axis=0).argmin())

        # compute sample variances for the leaf node
        HX = HX.var(axis=0)[l]
        HY = HY.var(axis=0)[l]

        HA = stein_hess(A, eta_G, eta_H).var(axis=0)[l] # TODO: compute this more efficiently

        vprint(f"l: {active_nodes[l]} Var_X = {HX} Var_Y = {HY} Var_Pool = {HA}")
        # store shifted nodes based on threshold and variance ratio
        if torch.min(HX, HY) * shifted_node_thres < HA:
            shifted_nodes.append(active_nodes[l])

        dict_stats[active_nodes[l]] = (HA / torch.min(HX, HY)).numpy()
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:, 0:l], X[:, l + 1:]])
        Y = torch.hstack([Y[:, 0:l], Y[:, l + 1:]])
    order.append(active_nodes[0])
    order.reverse()
    dict_stats = dict(sorted(dict_stats.items(), key=lambda item: -item[1]))
    
    if elbow:
        # if using the elbow heuristic, replace the list of shifted nodes
        shifted_nodes = _find_elbow(dict_stats, elbow_thres, elbow_online)
        
    return shifted_nodes, order, dict_stats


def test():
    from .utils import DataGenerator, node_metrics, set_seed
    set_seed(19)
    d, s0 = 20, 40
    generator = DataGenerator(d, s0, "ER")
    X, Y = generator.sample(1000, num_shifted_nodes=4)
    shifted_nodes, order = generator.shifted_nodes, np.arange(d)
    est_shifted_nodes, _, _ = est_node_shifts(X, Y, elbow=False)
    print(node_metrics(shifted_nodes, est_shifted_nodes, d))


if __name__ == "__main__":
    test()