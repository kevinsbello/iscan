:py:func:`iscan.score_estimator.stein_hess <iscan.score_estimator.stein_hess>`
==============================================================================
.. _iscan.score_estimator.stein_hess:
.. py:function:: iscan.score_estimator.stein_hess(X: torch.Tensor, eta_G: float, eta_H: float, s: Optional[float] = None) -> torch.Tensor

   Estimates the diagonal of the Hessian of :math:`\log p(x)` at the provided samples points :math:`X`,
   using first and second-order Stein identities.

   :param X: dataset X
   :type X: torch.Tensor
   :param eta_G: Coefficient of the L2 regularizer for estimation of the score.
   :type eta_G: float
   :param eta_H: Coefficient of the L2 regularizer for estimation of the score's Jacobian diagonal.
   :type eta_H: float
   :param s: Scale for the Kernel. If ``None``, the scale is estimated from data, by default ``None``.
   :type s: float, optional

   :returns: Estimation of the score's Jacobian diagonal.
   :rtype: torch.Tensor



