:py:meth:`iscan.utils.DataGenerator._delete_edges <iscan.utils.DataGenerator._delete_edges>`
============================================================================================
.. _iscan.utils.DataGenerator._delete_edges:
.. py:method:: iscan.utils.DataGenerator._delete_edges(adj: numpy.ndarray, shifted_nodes: numpy.ndarray) -> numpy.ndarray

   Generates a new adjacency matrix where some incoming edges (parents) are randomly deleted for the shifted nodes.

   :param adj: Base adjacency matrix
   :type adj: np.ndarray
   :param shifted_nodes: Vector of shifted node indices
   :type shifted_nodes: np.ndarray

   :returns: [d, d] adjacency matrix with removed edges
   :rtype: np.ndarray



