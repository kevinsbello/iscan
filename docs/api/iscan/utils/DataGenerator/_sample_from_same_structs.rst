:py:meth:`iscan.utils.DataGenerator._sample_from_same_structs <iscan.utils.DataGenerator._sample_from_same_structs>`
====================================================================================================================
.. _iscan.utils.DataGenerator._sample_from_same_structs:
.. py:method:: iscan.utils.DataGenerator._sample_from_same_structs(adj: numpy.ndarray, shifted_nodes: numpy.ndarray, use_gp: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]

   Sample data where both DAGs have the same structure

   :param adj: Adjacency matrix
   :type adj: np.ndarray
   :param shifted_nodes: Set of shited nodes
   :type shifted_nodes: np.ndarray
   :param use_gp: Whether to sample functions from Gaussian Processes, by default ``False``
   :type use_gp: bool, optional

   :returns: Datasets X and Y
   :rtype: Tuple[np.ndarray, np.ndarray]



