:py:func:`iscan.shifted_nodes._get_min_rank_sum <iscan.shifted_nodes._get_min_rank_sum>`
========================================================================================
.. _iscan.shifted_nodes._get_min_rank_sum:
.. py:function:: iscan.shifted_nodes._get_min_rank_sum(HX: torch.Tensor, HY: torch.Tensor) -> int

   | Find which node has the mininum rank sum in datasets X and dataset Y.
   | This is helpful to select a common leaf across the datasets.

   :param HX: Hessian's diagonal estimation for dataset X.
   :type HX: torch.Tensor
   :param HY: Hessian's diagonal estimation for dataset Y.
   :type HY: torch.Tensor

   :returns: Node index that has the smallest rank sum.
   :rtype: int



