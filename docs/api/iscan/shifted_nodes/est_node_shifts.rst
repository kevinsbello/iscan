:py:func:`iscan.shifted_nodes.est_node_shifts <iscan.shifted_nodes.est_node_shifts>`
====================================================================================
.. _iscan.shifted_nodes.est_node_shifts:
.. py:function:: iscan.shifted_nodes.est_node_shifts(X: Union[numpy.ndarray, torch.Tensor], Y: Union[numpy.ndarray, torch.Tensor], eta_G: float = 0.001, eta_H: float = 0.001, normalize_var: bool = False, shifted_node_thres: float = 2.0, elbow: bool = False, elbow_thres: float = 30.0, elbow_online: bool = True, use_both_rank: bool = True, verbose: bool = False) -> Tuple[list, list, dict]

   | Implementation of the iSCAN method of Chen et al. (2023).
   | Returns an estimated topological ordering, and estimated shifted nodes

   .. rubric:: References

   - T. Chen, K. Bello, B. Aragam, P. Ravikumar. (2023).
   `iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models. <https://arxiv.org/abs/2306.17361>`_.

   :param X: Dataset with shape :math:`(n,d)`, where :math:`n` is the number of samples,
             and :math:`d` is the number of variables/nodes.
   :type X: Union[np.ndarray, torch.Tensor]
   :param Y: Dataset with shape :math:`(n,d)`, where :math:`n` is the number of samples,
             and :math:`d` is the number of variables/nodes.
   :type Y: Union[np.ndarray, torch.Tensor]
   :param eta_G: hyperparameter for the score's Jacobian estimation, by default 0.001.
   :type eta_G: float, optional
   :param eta_H: hyperparameter for the score's Jacobian estimation, by default 0.001.
   :type eta_H: float, optional
   :param normalize_var: If ``True``, the Hessian's diagonal is normalized by the expected value, by default ``False``.
   :type normalize_var: bool, optional
   :param shifted_node_thres: Threshold to decide whether or not a variable has a distribution shift, by default 2.
   :type shifted_node_thres: float, optional
   :param elbow: If ``True``, iscan uses the elbow heuristic to determine shifted nodes. By default ``True``.
   :type elbow: bool, optional
   :param elbow_thres: If using the elbow method, ``elbow_thres`` is the ``hard_thres`` in
                       :py:func:`~iscan-dag.shifted_nodes.find_elbow` function, by default 30.
   :type elbow_thres: float, optional
   :param elbow_online: If using the elbow method, ``elbow_online`` is ``online`` in
                        :py:func:`~iscan-dag.shifted_nodes.find_elbow` function, by default ``True``.
   :type elbow_online: bool, optional
   :param use_both_rank: estimate topo order by X's and Y's rank sum. If False, only use X for topo order, by default ``False``.
   :type use_both_rank: bool, optional
   :param verbose: If ``True``, prints to stdout the variances of the Hessian entries for the running leafs.
                   By default ``False``.
   :type verbose: bool, optional

   :returns: estimated shifted nodes, topological order, and dict of variance ratios for all nodes.
   :rtype: Tuple[list, list, dict]



