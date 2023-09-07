import torch

def Stein_hess(X, eta_G, eta_H, s=None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
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