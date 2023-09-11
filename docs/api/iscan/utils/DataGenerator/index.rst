:py:class:`iscan.utils.DataGenerator <iscan.utils.DataGenerator>`
=================================================================

.. _iscan.utils.DataGenerator:

.. py:class:: iscan.utils.DataGenerator(d: int, s0: int, graph_type: str, noise_std: Union[float, numpy.ndarray, List[float]] = 0.5, noise_type: str = 'Gaussian')


   Generate synthetic data to evaluate shifted nodes and shifted edges

   Defines the class of graphs and data to be generated.

   :param d: Number of variables
   :type d: int
   :param s0: Expected number of edges in the random graphs
   :type s0: int
   :param graph_type: One of ``["ER", "SF"]``. ``ER`` and ``SF`` refer to Erdos-Renyi and Scale-free graphs, respectively.
   :type graph_type: str
   :param noise_std: Sets the scale (variance) of the noise distributions. Should be either a scalar, or a list (array) of dim d. By default 1.
   :type noise_std: Union[float, np.ndarray, List[float]], optional
   :param noise_type: One of ``["Gaussian", "Laplace", "Gumbel"]``, by default "Gaussian".
   :type noise_type: str, optional

   Methods
   ~~~~~~~

   .. autoapisummary::

      iscan.utils.DataGenerator._simulate_dag
      iscan.utils.DataGenerator._sample_GP
      iscan.utils.DataGenerator._choose_shifted_nodes
      iscan.utils.DataGenerator._delete_edges
      iscan.utils.DataGenerator._sample_from_same_structs
      iscan.utils.DataGenerator._sample_from_diff_structs
      iscan.utils.DataGenerator.sample

.. toctree::
   :maxdepth: 2
   :hidden:

   _simulate_dag<_simulate_dag>
   _sample_GP<_sample_GP>
   _choose_shifted_nodes<_choose_shifted_nodes>
   _delete_edges<_delete_edges>
   _sample_from_same_structs<_sample_from_same_structs>
   _sample_from_diff_structs<_sample_from_diff_structs>
   sample<sample>

