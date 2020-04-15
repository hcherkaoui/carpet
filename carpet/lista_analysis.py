""" Module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .lista_base import ListaBase, DOC_LISTA
from .checks import check_tensor
from .proximity import pseudo_soft_th_tensor


class StepSubGradTV(ListaBase):
    __doc__ = DOC_LISTA.format(type='learned-TV step',
                               problem_name='TV',
                               descr='only learn a step size'
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="one_shot", initial_parameters=[],
                 name="learned-TV Sub Gradient", ctx=None, verbose=0,
                 device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter,
                         per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        init_step_size = 1e-10

        parameters_config = dict(step_size=[], threshold=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0)
                layer_params['step_size'] = np.array(init_step_size)

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        x, output_layer = self._check_forward_inputs(
                                            x, output_layer, enable_none=True)

        Lz_hat = x

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = layer_params['step_size']
            mul_lbda = layer_params.get('threshold', 1.0)

            # apply one 'iteration'
            if Lz_hat is None:  # equivalent to initialized to Lz_hat = 0
                Lz_hat = step_size * x
            else:
                residual = Lz_hat - x
                reg = Lz_hat.matmul(self.D_).sign().matmul(self.D_.t())
                grad = residual + (lbda * mul_lbda) * reg
                Lz_hat = Lz_hat - step_size * grad

        return Lz_hat

    def transform(self, x, lbda, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            return self(x, lbda, output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]

        x = check_tensor(x, device=self.device)
        Lz_hat = check_tensor(z_hat, device=self.device)

        residual = Lz_hat - x
        loss = 0.5 * (residual * residual).sum()
        reg = lbda * torch.abs(Lz_hat.matmul(self.D_)).sum()
        return (loss + reg) / n_samples


class LChambolleTV(ListaBase):
    __doc__ = DOC_LISTA.format(
                type='learned-TV Chambolle original',
                problem_name='TV',
                descr='original parametrization from Gregor and Le Cun (2010)'
                )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="recursive", initial_parameters=[],
                 name="learned-TV Chambolle original", ctx=None, verbose=0,
                 device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter, per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        DtD = self.D.T.dot(self.D)
        Ik = np.eye(DtD.shape[0])

        parameters_config = dict(threshold=[], Wx=[], Wz=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0)
                layer_params['Wz'] = Ik - DtD / self.L
                layer_params['Wx'] = self.D / self.L

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        x, output_layer = self._check_forward_inputs(
                                            x, output_layer, enable_none=True)

        v_hat = None  # v0 = 0  # implicitly

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0)
            Wx = layer_params['Wx']
            Wz = layer_params['Wz']
            x_ = x / (lbda * mul_lbda)

            # apply one 'dual iteration'
            if v_hat is None:  # equivalent to initialized to v_hat = 0
                v_hat = x_.matmul(Wx)
            else:
                v_hat = v_hat.matmul(Wz) + x_.matmul(Wx)
                v_hat = torch.clamp(v_hat, -1.0, 1.0)

        return v_hat

    def transform(self, x, lbda, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            v_hat = self(x, lbda, output_layer=output_layer).cpu().numpy()
            return x - lbda * v_hat.dot(self.D.T)

    def _loss_fn(self, x, lbda, v_hat):
        """ Target loss function. """
        n_samples = x.shape[0]

        x = check_tensor(x, device=self.device)
        v_hat = check_tensor(v_hat, device=self.device)
        v_hat = torch.clamp(v_hat, -1.0, 1.0)

        if (torch.abs(v_hat) <= 1.0).all():
            residual = v_hat.matmul(self.D_.t()) - x / lbda  # dual formulation
            return 0.5 * (residual * residual).sum() / n_samples
        else:
            return torch.tensor([np.inf])


class CoupledChambolleTV(ListaBase):
    __doc__ = DOC_LISTA.format(type='learned-TV Chambolle-Coupled',
                               problem_name='TV',
                               descr=('one weight parametrization from Chen '
                                      'et al (2018)'),
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="recursive", initial_parameters=[],
                 name="learned-TV Chambolle-Coupled", ctx=None, verbose=0,
                 device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter,
                         per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        parameters_config = dict(threshold=[], W_coupled=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0)
                layer_params['W_coupled'] = self.D / self.L

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        x, output_layer = self._check_forward_inputs(
                                            x, output_layer, enable_none=True)

        v_hat = None  # v0 = 0  # implicitly

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            W = layer_params['W_coupled']
            mul_lbda = layer_params.get('threshold', 1.0)
            x_ = x / (lbda * mul_lbda)

            # apply one 'dual iteration'
            if v_hat is None:  # equivalent to initialized to v_hat = 0
                v_hat = x_.matmul(W)
            else:
                residual = v_hat.matmul(self.D_.t()) - x_
                v_hat = v_hat - residual.matmul(W)
                v_hat = torch.clamp(v_hat, -1.0, 1.0)

        return v_hat

    def transform(self, x, lbda, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            v_hat = self(x, lbda, output_layer=output_layer).cpu().numpy()
            return x - lbda * v_hat.dot(self.D.T)

    def _loss_fn(self, x, lbda, v_hat):
        """ Target loss function. """
        n_samples = x.shape[0]

        x = check_tensor(x, device=self.device)
        v_hat = check_tensor(v_hat, device=self.device)
        v_hat = torch.clamp(v_hat, -1.0, 1.0)

        if (torch.abs(v_hat) <= 1.0).all():
            residual = v_hat.matmul(self.D_.t()) - x / lbda  # dual formulation
            return 0.5 * (residual * residual).sum() / n_samples
        else:
            return torch.tensor([np.inf])


class StepChambolleTV(ListaBase):
    __doc__ = DOC_LISTA.format(type='learned-TV Chambolle-Step',
                               problem_name='TV',
                               descr=('only learn a step size for the '
                                      'Chambolle dual algorithm'),
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="one_shot", initial_parameters=[],
                 name="learned-TV Chambolle-Step", ctx=None, verbose=0,
                 device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter,
                         per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        parameters_config = dict(step_size=[], threshold=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0)
                layer_params['step_size'] = np.array(1.0 / self.L)

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # fixer correctement les init
        x, output_layer = self._check_forward_inputs(
                                             x, output_layer, enable_none=True)

        v_hat = None  # v0 = 0  # implicitly

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = layer_params['step_size']
            mul_lbda = layer_params.get('threshold', 1.0)
            W = self.D_ * step_size
            x_ = x / (lbda * mul_lbda)

            # apply one 'dual iteration'
            if v_hat is None:  # equivalent to initialized to v_hat = 0
                v_hat = x_.matmul(W)
            else:
                residual = v_hat.matmul(self.D_.t()) - x_
                v_hat = v_hat - residual.matmul(W)
                v_hat = torch.clamp(v_hat, -1.0, 1.0)

        return v_hat

    def transform(self, x, lbda, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            v_hat = self(x, lbda, output_layer=output_layer).cpu().numpy()
            return x - lbda * v_hat.dot(self.D.T)

    def _loss_fn(self, x, lbda, v_hat):
        """ Target loss function. """
        n_samples = x.shape[0]

        x = check_tensor(x, device=self.device)
        v_hat = check_tensor(v_hat, device=self.device)
        v_hat = torch.clamp(v_hat, -1.0, 1.0)

        if (torch.abs(v_hat) <= 1.0).all():
            residual = v_hat.matmul(self.D_.t()) - x / lbda  # dual formulation
            return 0.5 * (residual * residual).sum() / n_samples
        else:
            return torch.tensor([np.inf])


class CoupledCondatVu(ListaBase):
    __doc__ = DOC_LISTA.format(type='learned-Condat-Vu-coupled',
                               problem_name='TV',
                               descr=('one weight parametrization from Chen '
                                      'et al (2018)'),
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="recursive", initial_parameters=[],
                 name="learned-Condat-Vu-coupled", ctx=None, verbose=0, device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter,
                         per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        parameters_config = dict(W_coupled=[], threshold=[])

        self.rho = 1.0
        self.sigma = 0.5
        L_D, L_I = self.L, 1.0  # lipschtiz constant
        self.tau = 1.0 / (L_I / 2.0 + self.sigma * L_D**2)

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0)
                layer_params['W_coupled'] = self.D

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # fixer correctement les init
        x, output_layer = self._check_forward_inputs(
                                             x, output_layer, enable_none=True)

        Lz_hat = Lz_hat_old = x
        n_samples, n_dim = x.shape[0], self.D.shape[1]
        v_hat = v_hat_old = torch.zeros(
                                    (n_samples, n_dim), dtype=torch.float64)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            W = layer_params['W_coupled']
            mul_lbda = layer_params.get('threshold', 1.0)

            # apply one 'primal and dual iteration'
            # primal descent
            Lz_hat_new = (Lz_hat_old - self.tau * (Lz_hat_old - x) -
                          self.tau * v_hat_old.matmul(W.t()))
            # dual ascent
            v_hat_ = (v_hat_old +
                    self.sigma * (2 * Lz_hat_new - Lz_hat_old).matmul(W))
            v_hat_new = v_hat_ - self.sigma * pseudo_soft_th_tensor(
                        v_hat_ / self.sigma, lbda * mul_lbda, 1.0 / self.sigma)
            # update
            Lz_hat = self.rho * Lz_hat_new + (1.0 - self.rho) * Lz_hat_old
            v_hat = self.rho * v_hat_new + (1.0 - self.rho) * v_hat_old
            # storing
            Lz_hat_old = Lz_hat
            v_hat_old = v_hat

        return Lz_hat

    def transform(self, x, lbda, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            return self(x, lbda, output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]

        x = check_tensor(x, device=self.device)
        Lz_hat = check_tensor(z_hat, device=self.device)

        residual = Lz_hat - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(Lz_hat.matmul(self.D_)).sum()
        return (loss + lbda * reg) / n_samples


class StepCondatVu(ListaBase):
    __doc__ = DOC_LISTA.format(type='learned-Condat-Vu-step',
                               problem_name='TV',
                               descr=('Primal and dual step learn'),
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="one_shot", initial_parameters=[],
                 name="learned-Condat-Vu-step", ctx=None, verbose=0, device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter,
                         per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        parameters_config = dict(sigma=[])

        self.rho = 1.0

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                layer_params['sigma'] = np.array(0.5)

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        # fixer correctement les init
        x, output_layer = self._check_forward_inputs(
                                             x, output_layer, enable_none=True)

        Lz_hat = Lz_hat_old = x
        n_samples, n_dim = x.shape[0], self.D.shape[1]
        v_hat = v_hat_old = torch.zeros(
                                    (n_samples, n_dim), dtype=torch.float64)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            sigma = layer_params['sigma']
            L_D, L_I = self.L, 1.0  # lipschtiz constant
            tau = 1.0 / (L_I / 2.0 + sigma * L_D**2)

            # apply one 'primal and dual iteration'
            # primal descent
            Lz_hat_new = (Lz_hat_old - tau * (Lz_hat_old - x) -
                          tau * v_hat_old.matmul(self.D_.t()))
            # dual ascent
            v_hat_ = (v_hat_old +
                      sigma * (2 * Lz_hat_new - Lz_hat_old).matmul(self.D_))
            v_hat_new = v_hat_ - sigma * pseudo_soft_th_tensor(
                         v_hat_ / sigma, lbda, 1.0 / sigma)
            # update
            Lz_hat = self.rho * Lz_hat_new + (1.0 - self.rho) * Lz_hat_old
            v_hat = self.rho * v_hat_new + (1.0 - self.rho) * v_hat_old
            # storing
            Lz_hat_old = Lz_hat
            v_hat_old = v_hat

        return Lz_hat

    def transform(self, x, lbda, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            return self(x, lbda, output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]

        x = check_tensor(x, device=self.device)
        Lz_hat = check_tensor(z_hat, device=self.device)

        residual = Lz_hat - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(Lz_hat.matmul(self.D_)).sum()
        return (loss + lbda * reg) / n_samples


ALL_LTV = dict(stepsubgradient=StepSubGradTV, lchambolle=LChambolleTV,
               coupledchambolle=CoupledChambolleTV,
               stepchambolle=StepChambolleTV, coupledcondatvu=CoupledCondatVu,
               stepcondatvu=StepCondatVu)
