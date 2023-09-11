:py:meth:`iscan.utils.DataGenerator._simulate_dag <iscan.utils.DataGenerator._simulate_dag>`
============================================================================================
.. _iscan.utils.DataGenerator._simulate_dag:
.. py:method:: iscan.utils.DataGenerator._simulate_dag(d: int, s0: int, graph_type: str) -> numpy.ndarray

   Simulate random DAG with some expected number of edges.

   :param d: num of nodes
   :type d: int
   :param s0: expected num of edges
   :type s0: int
   :param graph_type: ER, SF
   :type graph_type: str

   :returns: :math:`(d, d)` binary adj matrix of a sampled DAG
   :rtype: np.ndarray



