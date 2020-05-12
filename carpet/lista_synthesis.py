""" Module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .lista_base import ListaBase, DOC_LISTA
from .proximity import pseudo_soft_th_tensor
from .checks import check_tensor
from .utils import init_vuz


class _ListaSynthesis(ListaBase):
    """Basis class for learn algorithms with synthesis formulation"""

    def transform(self, x, lbda, output_layer=None):
        with torch.no_grad():
            return self(x, lbda, output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z = check_tensor(z, device=self.device)
        residual = z.matmul(self.L_).matmul(self.A_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z[:, 1:]).sum()
        return (loss + lbda * reg) / n_samples


class ListaLASSO(_ListaSynthesis):
    __doc__ = DOC_LISTA.format(
        type='learned-ISTA original',
        problem_name='LASSO',
        descr='original parametrization from Gregor and Le Cun (2010)'
    )

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="LISTA", verbose=0, device=None):

        n_atoms = A.shape[0]

        self.A = np.array(A)
        self.L = np.triu(np.ones((n_atoms, n_atoms)))

        self.A_ = check_tensor(self.A, device=device)
        self.L_ = check_tensor(self.L, device=device)

        self.LA = self.L.dot(self.A)
        self.l_ = np.linalg.norm(self.LA.T, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        n_atoms = self.A.shape[0]
        I_k = np.eye(n_atoms)
        layer_params = dict()
        layer_params['Wz'] = I_k - self.LA.dot(self.LA.T) / self.l_
        layer_params['Wx'] = self.LA.T / self.l_
        if self.learn_th:
            layer_params['threshold'] = np.array(1.0)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # check inputs
        x, output_layer = self._check_forward_inputs(
            x, output_layer, enable_none=True
        )

        # initialized variables
        n_atoms = self.A.shape[0]
        D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]
        _, _, z = init_vuz(self.A, D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = 1.0 / self.l_
            mul_lbda = layer_params.get('threshold', 1.0)
            Wx = layer_params['Wx']
            Wz = layer_params['Wz']

            # apply one 'iteration'
            z = z.matmul(Wz) + x.matmul(Wx)
            z = pseudo_soft_th_tensor(z, lbda * step_size, mul_lbda)

        return z


class CoupledIstaLASSO(_ListaSynthesis):
    __doc__ = DOC_LISTA.format(
        type='learned-ISTA coupled',
        problem_name='LASSO',
        descr='one weight parametrization from Chen et al (2018)'
    )

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="Coupled-LISTA", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.L = np.triu(np.ones((n_atoms, n_atoms)))

        self.A_ = check_tensor(self.A, device=device)
        self.L_ = check_tensor(self.L, device=device)

        self.LA = self.L.dot(self.A)
        self.l_ = np.linalg.norm(self.LA.T, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        layer_params = dict()
        layer_params['W_coupled'] = self.LA.T / self.l_
        if self.learn_th:
            layer_params['threshold'] = np.array(1.0)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # check inputs
        x, output_layer = self._check_forward_inputs(
            x, output_layer, enable_none=True
        )

        # initialized variables
        n_atoms = self.A.shape[0]
        D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]
        _, _, z = init_vuz(self.A, D, x, lbda)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = 1.0 / self.l_
            mul_lbda = layer_params.get('threshold', 1.0)
            W = layer_params['W_coupled']

            # apply one 'iteration'
            residual = z.matmul(self.L_).matmul(self.A_) - x
            z = z - residual.matmul(W)
            z = pseudo_soft_th_tensor(z, lbda * step_size, mul_lbda)

        return z


class StepIstaLASSO(_ListaSynthesis):
    __doc__ = DOC_LISTA.format(
        type='learned-ISTA step',
        problem_name='LASSO',
        descr='only learn a step size'
    )

    def __init__(self, A, n_layers, learn_th=False, max_iter=100,
                 net_solver_type='recursive', initial_parameters=[],
                 name="Step-LISTA", verbose=0, device=None):

        n_atoms = A.shape[0]

        self.A = np.array(A)
        self.L = np.triu(np.ones((n_atoms, n_atoms)))

        self.A_ = check_tensor(self.A, device=device)
        self.L_ = check_tensor(self.L, device=device)

        self.LA = self.L.dot(self.A)
        self.l_ = np.linalg.norm(self.LA.T, ord=2) ** 2

        if learn_th:
            print("In StepIstaLASSO learn_th can't be enable, ignore it.")

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        """ Initialize the parameters of the network. """
        return dict(step_size=np.array(1.0 / self.l_))

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # check inputs
        x, output_layer = self._check_forward_inputs(
            x, output_layer, enable_none=True
        )

        # initialized variables
        n_atoms = self.A.shape[0]
        D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]
        _, _, z = init_vuz(self.A, D, x, lbda)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = layer_params.get('step_size', 1.0)
            W = self.A_.t().matmul(self.L_.t()) * step_size

            # apply one 'iteration'
            residual = z.matmul(self.L_).matmul(self.A_) - x
            z = z - residual.matmul(W)
            z = pseudo_soft_th_tensor(z, lbda * step_size, 1.0)

        return z
