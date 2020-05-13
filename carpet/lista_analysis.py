""" Module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .lista_base import ListaBase, DOC_LISTA
from .lista_synthesis import ListaLASSO
from .checks import check_tensor
from .proximity import pseudo_soft_th_tensor
from .utils import init_vuz, v_to_u
from .proximity_tv import ProxTV_l1


class _ListaAnalysis(ListaBase):

    def transform(self, x, lbda, output_layer=None):
        with torch.no_grad():
            return self(x, lbda, output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z = check_tensor(z, device=self.device)
        residual = z.matmul(self.A_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = lbda * torch.abs(z[:, 1:] - z[:, :-1]).sum()
        return (loss + reg) / n_samples


class _ListaAnalysisDual(ListaBase):

    def transform(self, x, lbda, output_layer=None):
        with torch.no_grad():
            v = self(x, lbda, output_layer=output_layer).cpu().numpy()
            return v_to_u(v, x, lbda, A=self.A, D=self.D, device=self.device)

    def _loss_fn(self, x, lbda, v):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        v = check_tensor(v, device=self.device)
        if (torch.abs(v) <= lbda).all():
            residual = v.matmul(self.Psi_A_.t())
            cost = 0.5 * (residual * residual).sum()
            cost += torch.diag(- x.matmul(self.Psi_A_).matmul(v.t())).sum()
            return cost / n_samples
        else:
            return torch.tensor([np.inf])


class StepSubGradTV(_ListaAnalysis):
    __doc__ = DOC_LISTA.format(type='learned-TV step',
                               problem_name='TV',
                               descr='only learn a step size'
                               )

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="learned-TV Sub Gradient", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)
        self.D_ = check_tensor(self.D, device=device)

        if learn_th:
            print("In StepSubGradTV learn_th can't be enable, ignore it.")

        super().__init__(n_layers=n_layers, learn_th=False,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        init_step_size = 1e-10
        return dict(step_size=np.array(init_step_size))

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # check inputs
        x, output_layer = self._check_forward_inputs(
            x, output_layer, enable_none=True
        )

        # initialized variables
        _, u, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = layer_params['step_size']

            # apply one 'iteration'
            residual = (u.matmul(self.A_) - x).matmul(self.A_.t())
            reg = u.matmul(self.D_).sign().matmul(self.D_.t())
            grad = residual + lbda * reg
            u = u - step_size * grad

        return u


class ListaTV(_ListaAnalysis):
    __doc__ = DOC_LISTA.format(
        type='learned-TV original',
        problem_name='TV',
        descr='original parametrization from Gregor and Le Cun (2010)'
    )

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="LISTA", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.I_k = np.eye(n_atoms)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)
        self.l_ = np.linalg.norm(self.A, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

        self.prox_tv = ListaLASSO(A=self.I_k, n_layers=500, learn_th=True,
                                  name="Prox-TV-Lista", device=self.device)

    def get_layer_parameters(self, layer):
        layer_params = dict()
        layer_params['Wu'] = self.I_k - self.A.dot(self.A.T) / self.l_
        layer_params['Wx'] = self.A.T / self.l_
        if self.learn_th:
            layer_params['threshold'] = np.array(1.0 / self.l_)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # check inputs
        x, output_layer = self._check_forward_inputs(
            x, output_layer, enable_none=True
        )

        # initialized variables
        _, u, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0 / self.l_)
            mul_lbda = check_tensor(mul_lbda, device=self.device)
            Wx = layer_params['Wx']
            Wu = layer_params['Wu']

            # apply one 'iteration'
            u = u.matmul(Wu) + x.matmul(Wx)
            z = self.prox_tv(x=u, lbda=lbda * mul_lbda)
            u = torch.cumsum(z, dim=1)

        return u


class OrigChambolleTV(_ListaAnalysisDual):
    __doc__ = DOC_LISTA.format(
        type='learned-TV Chambolle original',
        problem_name='TV',
        descr='original parametrization from Gregor and Le Cun (2010)'
    )

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="learned-TV Chambolle original", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)
        self.D_ = check_tensor(self.D, device=device)

        self.Psi_A = np.linalg.pinv(self.A).dot(self.D)
        self.Psi_AtPsi_A = self.Psi_A.T.dot(self.Psi_A)

        self.Psi_A_ = check_tensor(self.Psi_A, device=device)
        self.Psi_AtPsi_A_ = check_tensor(self.Psi_AtPsi_A, device=device)

        self.l_ = np.linalg.norm(self.Psi_A, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        n_atoms = self.A.shape[0]
        I_k = np.eye(n_atoms)
        layer_params = dict()
        layer_params['Wv'] = I_k - self.Psi_AtPsi_A / self.l_
        layer_params['Wx'] = self.Psi_A / self.l_
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
        v, _, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0)
            mul_lbda = check_tensor(mul_lbda)
            Wx = layer_params['Wx']
            Wv = layer_params['Wv']

            # apply one 'dual iteration'
            v = v.matmul(Wv) + x.matmul(Wx)
            v = torch.clamp(v, -lbda * mul_lbda, lbda * mul_lbda)

        return v


class CoupledChambolleTV(_ListaAnalysisDual):
    __doc__ = DOC_LISTA.format(
        type='learned-TV Chambolle-Coupled',
        problem_name='TV',
        descr='one weight parametrization from Chen et al (2018)'
    )

    def __init__(self, A, n_layers, learn_th=True,  max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="learned-TV Chambolle-Coupled", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)

        self.Psi_A = np.linalg.pinv(self.A).dot(self.D)
        self.Psi_AtPsi_A = self.Psi_A.T.dot(self.Psi_A)

        self.Psi_A_ = check_tensor(self.Psi_A, device=device)
        self.Psi_AtPsi_A_ = check_tensor(self.Psi_AtPsi_A, device=device)

        self.l_ = np.linalg.norm(self.Psi_A, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        layer_params = dict()
        layer_params['W_coupled'] = self.Psi_A / self.l_
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
        v, _, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)
        v = check_tensor(v, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            W = layer_params['W_coupled']
            mul_lbda = layer_params.get('threshold', 1.0)
            mul_lbda = check_tensor(mul_lbda, device=self.device)

            # apply one 'dual iteration'
            residual = v.matmul(self.Psi_A_.t()) - x
            v = v - residual.matmul(W)
            v = torch.clamp(v, -lbda * mul_lbda, lbda * mul_lbda)

        return v


class StepChambolleTV(_ListaAnalysisDual):
    __doc__ = DOC_LISTA.format(
        type='learned-TV Chambolle-Step',
        problem_name='TV',
        descr='only learn a step size for the Chambolle dual algorithm',
    )

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="learned-TV Chambolle-Step", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)

        self.Psi_A = np.linalg.pinv(self.A).dot(self.D)
        self.Psi_AtPsi_A = self.Psi_A.T.dot(self.Psi_A)

        self.Psi_A_ = check_tensor(self.Psi_A, device=device)
        self.Psi_AtPsi_A_ = check_tensor(self.Psi_AtPsi_A, device=device)

        self.l_ = np.linalg.norm(self.Psi_A, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        layer_params = dict(step_size=1 / self.l_)
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
        v, _, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = layer_params['step_size']
            mul_lbda = layer_params.get('threshold', 1.0)
            mul_lbda = check_tensor(mul_lbda, device=self.device)
            W = self.Psi_A_ * step_size

            # apply one 'dual iteration'
            residual = v.matmul(self.Psi_A_.t()) - x
            v = v - residual.matmul(W)
            v = torch.clamp(v, -lbda * mul_lbda, lbda * mul_lbda)

        return v


class CoupledCondatVu(_ListaAnalysis):
    __doc__ = DOC_LISTA.format(
        type='learned-Condat-Vu-coupled',
        problem_name='TV',
        descr='one weight parametrization from Chen et al (2018)'
    )

    def __init__(self, A, n_layers, learn_th=True, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="learned-Condat-Vu-coupled", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)
        self.D_ = check_tensor(self.D, device=device)

        self.l_A = np.linalg.norm(self.A, ord=2) ** 2
        self.l_D = np.linalg.norm(self.D, ord=2) ** 2

        # Condat-Vu parameters
        self.rho = 1.0
        self.sigma = 0.5
        self.tau = 1.0 / (self.l_A / 2.0 + self.sigma * self.l_D**2)

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        layer_params = dict(W_coupled=self.D)
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
        v, u, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)
        v_old, u_old, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            W = layer_params['W_coupled']
            mul_lbda = layer_params.get('threshold', 1.0)
            mul_lbda = check_tensor(mul_lbda, device=self.device)
            sigma = self.sigma
            tau = self.tau
            rho = self.rho

            # primal descent
            u_new = u_old + (
                - tau * (u_old.matmul(self.A_) - x).matmul(self.A_.t())
                - tau * v_old.matmul(W.t()))
            # dual ascent
            v_ = v_old + sigma * (2 * u_new - u_old).matmul(W)
            v_new = v_ - sigma * pseudo_soft_th_tensor(
                v_ / sigma, lbda * mul_lbda, 1.0 / sigma)
            # update
            u = rho * u_new + (1.0 - rho) * u_old
            v = rho * v_new + (1.0 - rho) * v_old
            # storing
            u_old = u
            v_old = v

        return u


class StepCondatVu(_ListaAnalysis):
    __doc__ = DOC_LISTA.format(
        type='learned-Condat-Vu-step',
        problem_name='TV',
        descr='Primal and dual step learn'
    )

    def __init__(self, A, n_layers, learn_th=False, max_iter=100,
                 net_solver_type="recursive", initial_parameters=[],
                 name="learned-Condat-Vu-step", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)
        self.D_ = check_tensor(self.D, device=device)

        self.l_A = np.linalg.norm(self.A, ord=2) ** 2
        self.l_D = np.linalg.norm(self.D, ord=2) ** 2

        # Parameter for accelerated condat-vu algorithm
        self.rho = 1.0

        if learn_th:
            print("In StepIstaLASSO learn_th can't be enable, ignore it.")

        super().__init__(n_layers=n_layers, learn_th=False,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        return dict(sigma=np.array(.5))

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # check inputs
        x, output_layer = self._check_forward_inputs(
            x, output_layer, enable_none=True
        )

        # initialized variables
        v, u, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)
        v_old, u_old, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            sigma = layer_params['sigma']
            sigma = torch.clamp(sigma, 0.5, 2.0)  # TODO constraint learning
            tau = 1.0 / (self.l_A / 2.0 + sigma * self.l_D**2)
            rho = self.rho

            # primal descent
            u_new = u_old + (
                - tau * (u_old.matmul(self.A_) - x).matmul(self.A_.t())
                - tau * v_old.matmul(self.D_.t()))
            # dual ascent
            v_ = v_old + sigma * (2 * u_new - u_old).matmul(self.D_)
            v_new = v_ - sigma * pseudo_soft_th_tensor(
                v_ / sigma, lbda, 1.0 / sigma)
            # update
            u = rho * u_new + (1.0 - rho) * u_old
            v = rho * v_new + (1.0 - rho) * v_old
            # storing
            u_old = u
            v_old = v

        return u


class LpgdTautString(_ListaAnalysis):
    __doc__ = DOC_LISTA.format(
        type='learned-PGD with taut-string for prox operator',
        problem_name='TV',
        descr='unconstrained parametrization'
    )

    def __init__(self, A, n_layers, learn_th=False, max_iter=100,
                 net_solver_type="recursive", initial_parameters=None,
                 name="LPGD analysis", verbose=0, device=None):

        n_atoms = A.shape[0]
        self.A = np.array(A)
        self.I_k = np.eye(n_atoms)
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.A_ = check_tensor(self.A, device=device)
        self.l_ = np.linalg.norm(self.A, ord=2) ** 2

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_layer_parameters(self, layer):
        layer_params = dict()
        layer_params['Wu'] = self.I_k - self.A.dot(self.A.T) / self.l_
        layer_params['Wx'] = self.A.T / self.l_
        if self.learn_th:
            layer_params['threshold'] = np.array(1.0 / self.l_)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # check inputs
        x, output_layer = self._check_forward_inputs(
            x, output_layer, enable_none=True
        )

        # initialized variables
        _, u, _ = init_vuz(self.A, self.D, x, lbda, device=self.device)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0 / self.l_)
            mul_lbda = check_tensor(mul_lbda, device=self.device)
            Wx = layer_params['Wx']
            Wu = layer_params['Wu']

            # apply one 'iteration'
            u = u.matmul(Wu) + x.matmul(Wx)
            u = ProxTV_l1.apply(u, lbda * mul_lbda)

        return u
