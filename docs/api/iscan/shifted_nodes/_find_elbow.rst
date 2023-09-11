:py:func:`iscan.shifted_nodes._find_elbow <iscan.shifted_nodes._find_elbow>`
============================================================================
.. _iscan.shifted_nodes._find_elbow:
.. py:function:: iscan.shifted_nodes._find_elbow(diff_dict: dict, hard_thres: float = 30, online: bool = True) -> numpy.ndarray

   Return selected shifted nodes by finding elbow point on sorted variance

   :param diff_dict: A dictionary where ``key`` is the index of variables/nodes, and ``value`` is its variance ratio.
   :type diff_dict: dict
   :param hard_thres: | Variance ratios larger than hard_thres will be directly regarded as shifted.
                      | Selected nodes in this step will not participate in elbow method, by default 30.
   :type hard_thres: float, optional
   :param online: If ``True``, the heuristic will find a more aggressive elbow point, by default ``True``.
   :type online: bool, optional

   :returns: A dict with selected nodes and corresponding variance
   :rtype: np.ndarray



