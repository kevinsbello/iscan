:py:meth:`iscan.utils.DataGenerator.sample <iscan.utils.DataGenerator.sample>`
==============================================================================
.. _iscan.utils.DataGenerator.sample:
.. py:method:: iscan.utils.DataGenerator.sample(n: int, num_shifted_nodes: int, change_struct: bool = False, use_gp: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]

   Samples two datasets from randomly generated DAGs

   :param n: Number of samples
   :type n: int
   :param num_shifted_nodes: Desired number of shifted nodes. Actual number can be lower.
   :type num_shifted_nodes: int
   :param change_struct: Whether or not to sample from DAGs with different structures, by default ``False``.
   :type change_struct: bool, optional
   :param use_gp: Whether or not to sample from Gaussian Processes, by default ``True``.
   :type use_gp: bool, optional

   :returns: Two datasets with shape [n, d]
   :rtype: Tuple[np.ndarray, np.ndarray]

   :raises NotImplementedError: The case of sampling from GPs when the structures are differents is not implemented.



