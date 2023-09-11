:py:func:`iscan.shifted_edges._estimate_parents_foci <iscan.shifted_edges._estimate_parents_foci>`
==================================================================================================
.. _iscan.shifted_edges._estimate_parents_foci:
.. py:function:: iscan.shifted_edges._estimate_parents_foci(X: numpy.ndarray, node_set: numpy.ndarray, order: numpy.ndarray) -> numpy.ndarray

   Uses FOCI [1] to estimate local parents of ``node_set`` given a topological order in ``order``.

   .. rubric:: References

   1. Azadkia and Chatterjee (2019),"A simple measure of conditional dependence" <arXiv:1910.12327>.

   :param X: Dataset [n, d]
   :type X: np.ndarray
   :param node_set: Set of nodes of interest
   :type node_set: np.ndarray
   :param order: Topological ordering
   :type order: np.ndarray

   :returns: Adjacency matrix [d,d] with estimation of local parents
   :rtype: np.ndarray



