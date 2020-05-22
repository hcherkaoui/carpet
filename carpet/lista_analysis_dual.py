""" Module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .checks import check_tensor
from .utils import init_vuz, v_to_u
from .lista_base import ListaBase, DOC_LISTA


class _ListaAnalysisDual(ListaBase):

    _output = 'v-analysis_dual'

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=None,
                 name=None, verbose=0, device=None):

        if name is None:
            name = self.default_name

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]
        inv_A = np.linalg.pinv(self.A)

        self.A_ = check_tensor(self.A, device=device)
        self.inv_A_ = check_tensor(inv_A, device=device)

        self.Psi_A = inv_A.dot(self.D)
        self.Psi_AtPsi_A = self.Psi_A.T.dot(self.Psi_A)

        self.Psi_A_ = check_tensor(self.Psi_A, device=device)
        self.Psi_AtPsi_A_ = check_tensor(self.Psi_AtPsi_A, device=device)

        self.l_ = np.linalg.norm(self.Psi_A, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def transform(self, x, lbda, output_layer=None):
        v = super().transform(x, lbda, output_layer=output_layer)
        return v_to_u(v, x, lbda, A=self.A, D=self.D, device=self.device)

    def _loss_fn(self, x, lbda, v):
        r"""Loss function for the dual.

            :math:`L(v) = 1/2 ||A^\dagger D v||_2^2 - x A^\dagger D v`

        """
        # Check feasibility of the point
        if (torch.abs(v) > lbda).any():
            return torch.tensor([np.inf])

        n_samples = x.shape[0]
        residual = v.matmul(self.Psi_A_.t())
        cost = 0.5 * (residual * residual).sum()
        cost -= (x.matmul(self.Psi_A_) * v.t()).sum()
        return cost / n_samples


class OrigChambolleTV(_ListaAnalysisDual):

    # Class variables
    default_name = 'learned-TV Chambolle original'
    __doc__ = DOC_LISTA.format(
        type=default_name,
        problem_name='TV',
        descr='original parametrization from Gregor and Le Cun (2010)'
    )

    def get_initial_layer_parameters(self, layer_id):
        # TODO: this is not n_atom but n_atom - 1. Should be checked
        n_atoms = self.D.shape[1]
        I_k = np.eye(n_atoms)
        layer_params = dict()
        layer_params['Wv'] = I_k - self.Psi_AtPsi_A / self.l_
        layer_params['Wx'] = self.Psi_A / self.l_

        # TODO: we actually cannot learn this parameter with the code as we use
        # clamp which does not support tensor for its threshold. Should we
        # fixed this?
        if self.learn_th:
            layer_params['threshold'] = np.array(1.0)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        output_layer = self.check_output_layer(output_layer)

        # initialized variables
        v, _, _ = init_vuz(self.A, self.D, x, lbda, inv_A=self.inv_A_,
                           device=self.device)

        for layer_id in range(output_layer):
            layer_params = self.parameter_groups[f'layer-{layer_id}']
            # retrieve parameters
            # mul_lbda = layer_params.get('threshold', 1.0)
            # mul_lbda = check_tensor(mul_lbda)
            Wx = layer_params['Wx']
            Wv = layer_params['Wv']

            # apply one 'dual iteration'
            v = v.matmul(Wv) + x.matmul(Wx)
            v = torch.clamp(v, -lbda, lbda)
            # v = torch.clamp(v, -lbda * mul_lbda, lbda * mul_lbda)

        return v


class CoupledChambolleTV(_ListaAnalysisDual):

    # Class variables
    default_name = "learned-TV Chambolle-Coupled"
    __doc__ = DOC_LISTA.format(
        type=default_name,
        problem_name='TV',
        descr='one weight parametrization from Chen et al (2018)'
    )

    def get_initial_layer_parameters(self, layer_id):
        layer_params = dict()
        layer_params['W_coupled'] = self.Psi_A / self.l_
        # if self.learn_th:
        #     layer_params['threshold'] = np.array(1.0)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        output_layer = self.check_output_layer(output_layer)

        # initialized variables
        v, _, _ = init_vuz(self.A, self.D, x, lbda, inv_A=self.inv_A_,
                           device=self.device)

        for layer_id in range(output_layer):
            layer_params = self.parameter_groups[f'layer-{layer_id}']
            # retrieve parameters
            W = layer_params['W_coupled']

            # apply one 'dual iteration'
            residual = v.matmul(self.Psi_A_.t()) - x
            v = v - residual.matmul(W)
            v = torch.clamp(v, -lbda, lbda)

        return v


class StepChambolleTV(_ListaAnalysisDual):

    # Class variables
    default_name = "learned-TV Chambolle-Step"
    __doc__ = DOC_LISTA.format(
        type=default_name,
        problem_name='TV',
        descr='only learn a step size for the Chambolle dual algorithm',
    )

    def get_initial_layer_parameters(self, layer_id):
        layer_params = dict(step_size=1 / self.l_)
        # if self.learn_th:
        #     layer_params['threshold'] = np.array(1.0)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        output_layer = self.check_output_layer(output_layer)

        # initialized variables
        v, _, _ = init_vuz(self.A, self.D, x, lbda, inv_A=self.inv_A_,
                           device=self.device)

        for layer_id in range(output_layer):
            layer_params = self.parameter_groups[f'layer-{layer_id}']
            # retrieve parameters
            step_size = layer_params['step_size']
            W = self.Psi_A_ * step_size

            # apply one 'dual iteration'
            residual = v.matmul(self.Psi_A_.t()) - x
            v = v - residual.matmul(W)
            v = torch.clamp(v, -lbda, lbda)

        return v
