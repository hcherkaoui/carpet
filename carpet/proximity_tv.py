""" Utils module for examples. """
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

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
        x = x.detach().cpu().data

        # The regularization can be learnable or a float
        if isinstance(lbda, torch.Tensor):
            lbda = lbda.detach().cpu().data

        # Get back a tensor for the output and save it for the backward pass
        output = check_tensor(
            np.array([prox_tv.tv1_1d(xx, lbda) for xx in x]),
            device=device, requires_grad=True,
        )
        z = output - torch.functional.F.pad(output, (1, 0))[..., :-1]
        ctx.save_for_backward(torch.sign(z))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of proxTV using implicit gradient."""
        batch_size, n_dim = grad_output.shape
        sign_z, = ctx.saved_tensors
        device = grad_output.device
        S = sign_z != 0
        S[:, 0] = True
        sign_z[:, 0] = 0
        # XXX do clever computations
        L = torch.triu(torch.ones((n_dim, n_dim), dtype=torch.float64,
                       device=device))

        grad_x, grad_lbda = [], []
        for i in range(batch_size):
            L_S = L[:, S[i]]                     # n_dim x |S|
            grad_u = grad_output[i].matmul(L_S)  # 1 x |S|
            H_S = torch.inverse(L_S.t().matmul(L_S))
            grad_x.append(grad_u.matmul(H_S.matmul(L_S.t())))
            grad_lbda.append(grad_u.matmul(H_S.matmul(-sign_z[i][S[i]])))
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
    def forward(ctx, loss, u, lbda):
        assert loss.ndim == 0, (
            "The regularized loss should be a scalar. "
            f"Got ndim={loss.ndim}"
        )

        # Compute the reg and store the value necessary to compute
        # the gradient. Return the loss.
        reg = torch.abs(u[:, 1:] - u[:, :-1]).sum()
        ctx.save_for_backward(loss, reg, u, lbda)
        return loss + lbda * reg

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of loss + mu*reg using a prox step.

        The gradient is derived as the additive update that would be
        used in a proximal gradient descent:

            G(u) = (u - prox(u - eps * nabla(loss)(u), eps*lbda))/eps

        with a small eps (here hard coded to 1e-10).
        """
        loss, reg, u, lbda = ctx.saved_tensors

        device = u.device

        # do clever computations
        eps = 1e-10
        grad, = torch.autograd.grad(loss, u, only_inputs=True,
                                    retain_graph=True)
        x = (u - eps * grad).data
        lbda = lbda.data

        prox_x = check_tensor(
            np.array([prox_tv.tv1_1d(xx, eps * lbda) for xx in x]),
            device=device,
        )
        grad_u = (u - prox_x) / eps
        grad_lbda = reg.clone()
        return (torch.ones(0), grad_u, grad_lbda)
