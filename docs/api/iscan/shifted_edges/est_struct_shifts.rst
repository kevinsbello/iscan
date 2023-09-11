:py:func:`iscan.shifted_edges.est_struct_shifts <iscan.shifted_edges.est_struct_shifts>`
========================================================================================
.. _iscan.shifted_edges.est_struct_shifts:
.. py:function:: iscan.shifted_edges.est_struct_shifts(X: numpy.ndarray, Y: numpy.ndarray, shifted_nodes: Union[list, numpy.ndarray], order: Union[list, numpy.ndarray], method: str = 'foci') -> numpy.ndarray

   Estimates structural changes from data for a subset of variables (shifted nodes),
   given an order of the variables.

   :param X: Dataset X
   :type X: np.ndarray
   :param Y: Dataset Y
   :type Y: np.ndarray
   :param shifted_nodes: Set of variables of interest
   :type shifted_nodes: Union[list, np.ndarray]
   :param order: Valid topological ordering of the variables
   :type order: Union[list, np.ndarray]
   :param method: method to use for estimation of local parents. One of ``["foci"]``.
   :type method: str, optional

   :returns: Estimation of structural changes for the set of shifted nodes
   :rtype: np.ndarray



