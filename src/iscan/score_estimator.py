import torch
from typing import Optional


def stein_hess(X: torch.Tensor, eta_G: float, eta_H: float, s: Optional[float] = None) -> torch.Tensor:
    r"""
    Estimates the diagonal of the Hessian of :math:`\log p(x)` at the provided samples points :math:`X`, 
    using first and second-order Stein identities.

    Parameters
    ----------
    X : torch.Tensor
        dataset X
    eta_G : float
        Coefficient of the L2 regularizer for estimation of the score.
    eta_H : float
        Coefficient of the L2 regularizer for estimation of the score's Jacobian diagonal.
    s : float, optional
        Scale for the Kernel. If ``None``, the scale is estimated from data, by default ``None``.

    Returns
    -------
    torch.Tensor
        Estimation of the score's Jacobian diagonal.
    """
    torch.set_default_dtype(torch.double)
    n, d = X.shape
    X_diff = X.unsqueeze(1) - X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s ** 2)) / s

    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s ** 2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)

    nabla2K = torch.einsum('kij,ik->kj', -1 / s ** 2 + X_diff ** 2 / s ** 4, K)
    return -G ** 2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)