import torch
import prox_tv
import numpy as np
from .checks import check_tensor
from .lista_synthesis import CoupledIstaLASSO


class _ProxTV_l1(torch.autograd.Function):
    """ Custom autograd Functions wrapper for the prox_tv. """
    @staticmethod
    def forward(ctx, z, mu):
        # Convert input to numpy array to use the prox_tv library
        device = z.device
        z = z.detach().numpy()
        mu.detach().numpy()
        # Get back a tensor for the output and save it for the backward pass
        output = check_tensor(
            np.array([prox_tv.tv1_1d(zz, mu) for zz in z]),
            device=device, requires_grad=True
        )
        mask_U = output - torch.functional.F.pad(output, (1, 0))[..., :-1]
        ctx.save_for_backward(torch.sign(mask_U))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """ Compute the gradient of proxTV. """
        batch_size, n_dim = grad_output.shape
        sign_u, = ctx.saved_tensors
        S = sign_u != 0
        sign_u[0] = 0
        # do clever computations
        L = torch.triu(torch.ones((n_dim, n_dim), dtype=torch.float64))
        grad_z, grad_mu = [], []
        for i in range(batch_size):
            L_S = L[:, S[i]]                    # n_dim x |S|
            grad_u = grad_output[i].matmul(L_S)  # 1 x |S|
            H_S = torch.inverse(L_S.t().matmul(L_S))
            grad_z.append(grad_u.matmul(H_S.matmul(L_S.t())))
            grad_mu.append(grad_u.matmul(H_S.matmul(L_S.t())))
        grad_z = torch.stack(grad_z)
        grad_mu = torch.stack(grad_mu)
        return (grad_z, grad_mu)


class ProxTV(torch.nn.Module):
    """ProxTV layer"""

    def __init__(self, prox='lista', n_dim=None, n_layers=None,
                 device=None):
        assert prox in {'lista', 'prox_tv'}, (
            f"Unknown parameter prox='{prox}'. "
            "Should be one of {'lista', 'prox_tv'}"
        )
        self.prox = prox
        super().__init__()
        if self.prox == 'lista':
            assert n_dim is not None
            assert n_layers is not None
            self.lista = CoupledIstaLASSO(
                A=np.eye(n_dim), n_layers=n_layers, device=device)

    def forward(self, z, mu):
        if self.prox == 'prox_tv':
            return _ProxTV_l1.apply(z, mu)
        output = self.lista(z, mu)
        return torch.cumsum(output, dim=1)

