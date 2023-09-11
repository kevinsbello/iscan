:py:func:`iscan.utils.ddag_metrics <iscan.utils.ddag_metrics>`
==============================================================
.. _iscan.utils.ddag_metrics:
.. py:function:: iscan.utils.ddag_metrics(adj_true: numpy.ndarray, adj_pred: numpy.ndarray) -> dict

   Computes precision, recall, and f1 scores for the predicted structural changes/shifts (difference DAG).

   :param adj_true: Adjancecy matrix of the true difference DAG (graph of structural differences)
   :type adj_true: np.ndarray
   :param adj_pred: Predicted adjancecy matrix of the difference DAG (graph of structural differences)
   :type adj_pred: np.ndarray

   :returns: Precision, recall, and F1 scores
   :rtype: dict



