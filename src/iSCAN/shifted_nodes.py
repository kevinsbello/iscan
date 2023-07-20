import pandas as pd
from kneed import KneeLocator
import numpy as np
import torch
from score_estimator import Stein_hess


def find_elbow(diff_dict,hard_thres = 30,online = True):
    """
    Return selected shifted nodes by finding elbow point on sorted variance
    :param diff_dict: dict. key is index of nodes, value is variance
    :param hard_thres: variance larger than hard_thres will be directly regarded
                        as shifted. Selected nodes in this step will not participate
                        in elbow method.
    :param online: True will find more aggressive elbow point.
    :return: A dict with selected nodes and corresponding variance
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

def get_min_rank_sum(HX,HY):
    """
    Find which node has the min rank sum in dataset X and dataset Y
    :param HX: Matrix. Hessian estimation for dataset X.
    :param HY: Matrix. Hessian estimation for dataset Y.
    :return: int. An index of node who has the smallest rank sum.
    """
    order_X = torch.argsort(HX.var(axis=0))
    rank_X = torch.argsort(order_X)

    order_Y = torch.argsort(HY.var(axis=0))
    rank_Y = torch.argsort(order_Y)
    l = int((rank_X + rank_Y).argmin())
    return l

def iSCAN(X, Y, eta_G, eta_H, normalize_var=False, shifted_thres=2,
                        elbow=True,elbow_thres=30,elbow_online=True,use_both_rank = False):
    """
    Return estimated topo order and estimated shifted nodes
    :param X: Dataset sampling from graph
    :param Y: Dataset sampling from graph
    :param eta_G: regularization term
    :param eta_H: regularization term
    :param normalize_var: Boolean. Normalize data by mean
    :param shifted_thres: A node whose variance ratio is larger than this thres will be regraded as shifted nodes.
                            Will be ignored if elbow is True.
    :param elbow: Boolean. Whether to find shifted nodes by elbow method.
    :param elbow_thres:If using elbow method, elbow_thres is hard_thres in find_elbow function.
    :param elbow_online: If using elbow method, elbow_online is online in find_elbow function.
    :param use_both_rank:estimate topo order by X's and Y's rank sum. If False, only use X for topo order.
    :return: topo order, estimated shifted nodes, and corresponding variance ratio.
    """
    n, d = X.shape
    order = []
    shift_node = []
    active_nodes = list(range(d))
    diff_dict = dict()

    HX_dict = dict()
    HY_dict = dict()
    HA_dict = dict()

    for i in range(d - 1):

        A = torch.concat((X, Y))
        HX = Stein_hess(X, eta_G, eta_H)
        HY = Stein_hess(Y, eta_G, eta_H)

        if not use_both_rank:
            if normalize_var:
                HX = HX / HX.mean(axis=0)# The one mentioned in the paper
            l = int(HX.var(axis=0).argmin())
        else:
            if normalize_var:
                HX = HX / HX.mean(axis=0)
                HY = HY / HY.mean(axis=0)
            l = get_min_rank_sum(HX,HY)

        HX = HX.var(axis=0)[l]
        HY = HY.var(axis=0)[l]
        HA = Stein_hess(A, eta_G, eta_H).var(axis=0)[l]

        #HX_dict[active_nodes[l]] = HX.numpy()
        #HY_dict[active_nodes[l]] = HY.numpy()
        #HA_dict[active_nodes[l]] = HA.numpy()

        if torch.min(HX, HY) * shifted_thres < HA:
            shift_node.append(active_nodes[l])

        diff_dict[active_nodes[l]] = (HA / torch.min(HX, HY)).numpy()
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:, 0:l], X[:, l + 1:]])
        Y = torch.hstack([Y[:, 0:l], Y[:, l + 1:]])
    order.append(active_nodes[0])
    order.reverse()
    diff_dict = dict(sorted(diff_dict.items(), key=lambda item: -item[1]))
    if elbow:
        shift_node = find_elbow(diff_dict,elbow_thres,elbow_online)
    return order, shift_node, diff_dict