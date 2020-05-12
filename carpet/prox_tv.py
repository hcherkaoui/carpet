import torch
import prox_tv
import numpy as np

from .checks import check_tensor
from .lista_synthesis import CoupledIstaLASSO


class ProxTV_l1(torch.autograd.Function):
    """
    Custom autograd Function wrapper for the prox_tv.
    """

    @staticmethod
    def forward(ctx, x, lbda):
        # Convert input to numpy array to use the prox_tv library
        device = x.device
        x = x.data
        lbda = lbda.data

        # Get back a tensor for the output and save it for the backward pass
        output = check_tensor(
            np.array([prox_tv.tv1_1d(xx, lbda) for xx in x]),
            device=device, requires_grad=True
        )
        mask_U = output - torch.functional.F.pad(output, (1, 0))[..., :-1]
        assert np.allclose(np.diff(output.detach().numpy()), mask_U[:, 1:])
        ctx.save_for_backward(torch.sign(mask_U))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of proxTV using implicit gradient."""
        batch_size, n_dim = grad_output.shape
        sign_u, = ctx.saved_tensors
        S = sign_u != 0
        S[:, 0] = True
        sign_u[:, 0] = 0
        # do clever computations
        L = torch.triu(torch.ones((n_dim, n_dim), dtype=torch.float64))

        # import ipdb; ipdb.set_trace()
        grad_x, grad_lbda = [], []
        for i in range(batch_size):
            L_S = L[:, S[i]]                    # n_dim x |S|
            grad_u = grad_output[i].matmul(L_S)  # 1 x |S|

            H_S = torch.inverse(L_S.t().matmul(L_S))
            grad_x.append(grad_u.matmul(H_S.matmul(L_S.t())))
            grad_lbda.append(grad_u.matmul(H_S.matmul(sign_u[i][S[i]])))
        grad_x = torch.stack(grad_x)
        grad_lbda = torch.stack(grad_lbda)
        return (grad_x, grad_lbda)


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

    def forward(self, x, lbda):
        if self.prox == 'prox_tv':
            return ProxTV_l1.apply(x, lbda)

        output = self.lista(x, lbda)
        return torch.cumsum(output, dim=1)


class RegTV(torch.autograd.Function):
    """
    Custom autograd Functions wrapper for a loss regularized with TV

    It is a in the classical loss except that it computes the gradient
    using the prox instead of using sub-derivatives.
    """

    @staticmethod
    def forward(ctx, loss, z, mu):
        assert loss.ndim == 1, (
            "The regularized loss should be of shape (batch_size,). "
            f"Got ndim={loss.ndim}"
        )

        # Compute the reg and store the value necessary to compute
        # the gradient. Return the loss.
        reg = torch.abs(z[:, 1:] - z[:, :-1]).sum(axis=1)
        ctx.save_for_backward(loss, reg, z, mu)
        return loss + mu * reg

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of loss + mu*reg using a prox step.

        The gradient is derived as the additive update that would be
        used in a proximal gradient descent:

            G(z) = (z - prox(z - eps * nabla(loss)(z), eps*mu))/eps

        with a small eps (here hard coded to 1e-10).
        """
        loss, reg, z, mu = ctx.saved_tensors

        device = z.device

        # do clever computations
        eps = 1e-10
        grad, = torch.autograd.grad(loss, z, only_inputs=True,
                                    retain_graph=True)
        x = (z - eps * grad).data
        mu = mu.data

        prox_x = check_tensor(
            np.array([prox_tv.tv1_1d(xx, eps * mu) for xx in x]),
            device=device
        )
        grad_z = (z - prox_x) / eps
        grad_mu = reg.clone()
        return (torch.ones(0), grad_z, grad_mu)
