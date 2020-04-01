""" Module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .lista_base import ListaBase, DOC_LISTA
from .proximity import soft_thresholding
from .utils import get_alista_weights
from .checks import check_tensor


class Lista(ListaBase):
    __doc__ = DOC_LISTA.format(
                type='L-ISTA original',
                problem_name='LASSO',
                descr='original parametrization from Gregor and Le Cun (2010)'
                )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="recursive", initial_parameters=[],
                 name="LISTA", ctx=None, verbose=1, device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter, per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        n_atoms = self.D.shape[0]
        I_k = np.eye(n_atoms)

        parameters_config = dict(threshold=[], Wx=[], Wz=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = {}
                if self.learn_th:
                    layer_params['threshold'] = np.ones(n_atoms) / self.L
                layer_params['Wz'] = I_k - self.B / self.L
                layer_params['Wx'] = self.D.T / self.L

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lmbd, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = np.zeros_like(np.array(x).dot(self.D.T)) if z0 is None else z0
        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(x, z0,
                                                            output_layer,
                                                            enable_none=False)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            th = layer_params['threshold']
            Wx = layer_params['Wx']
            Wz = layer_params['Wz']

            # apply one 'iteration'
            z_hat = (z_hat.matmul(Wz) + x.matmul(Wx))
            z_hat = soft_thresholding(z_hat, lmbd, th)

        return z_hat

    def _loss_fn(self, x, lmbd, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat).sum()
        return (loss + lmbd * reg) / n_samples


class CoupledLista(ListaBase):
    __doc__ = DOC_LISTA.format(
                    type='L-ISTA coupled',
                    problem_name='LASSO',
                    descr='one weight parametrization from Chen et al (2018)'
                    )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="recursive", initial_parameters=[],
                 name="Coupled-LISTA", ctx=None, verbose=1, device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter, per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        n_atoms = self.D.shape[0]

        parameters_config = dict(threshold=[], W_coupled=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = {}
                if self.learn_th:
                    layer_params['threshold'] = np.ones(n_atoms) / self.L
                layer_params['W_coupled'] = self.D.T / self.L

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lmbd, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = np.zeros_like(np.array(x).dot(self.D.T)) if z0 is None else z0
        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(x, z0,
                                                            output_layer,
                                                            enable_none=False)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            th = layer_params['threshold']
            step_size = layer_params.get('step_size', 1.0)
            W = layer_params['W_coupled'] * step_size

            # apply one 'iteration'
            residual = z_hat.matmul(self.D_) - x
            z_hat = z_hat - residual.matmul(W)
            z_hat = soft_thresholding(z_hat, lmbd, th)

        return z_hat

    def _loss_fn(self, x, lmbd, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat).sum()
        return (loss + lmbd * reg) / n_samples


class ALista(ListaBase):
    __doc__ = DOC_LISTA.format(
                            type='L-ISTA alista',
                            problem_name='LASSO',
                            descr='analytic weights from Chen et al (2019)'
                            )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="recursive", initial_parameters=[],
                 name="ALISTA", ctx=None, verbose=1, device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter, per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        self.W = check_tensor(get_alista_weights(self.D).T, device=self.device)

        n_atoms = self.D.shape[0]

        parameters_config = dict(threshold=[], step_size=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = {}
                if self.learn_th:
                    layer_params['threshold'] = np.ones(n_atoms) / self.L
                layer_params['step_size'] = np.array(1 / self.L)
                layer_params['threshold'] = np.array(1 / self.L)

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lmbd, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = np.zeros_like(np.array(x).dot(self.D.T)) if z0 is None else z0
        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(x, z0,
                                                            output_layer,
                                                            enable_none=False)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            th = layer_params['threshold']
            step_size = layer_params.get('step_size', 1.)
            W = self.W * step_size

            # apply one 'iteration'
            residual = z_hat.matmul(self.D_) - x
            z_hat = z_hat - residual.matmul(W)
            z_hat = soft_thresholding(z_hat, lmbd, th)

        return z_hat

    def _loss_fn(self, x, lmbd, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat).sum()
        return (loss + lmbd * reg) / n_samples


class HessianLista(ListaBase):
    __doc__ = DOC_LISTA.format(
                type='L-ISTA hessian',
                problem_name='LASSO',
                descr='one weight parametrization as a quasi newton technique'
                            )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="recursive", initial_parameters=[],
                 name="Hessian-LISTA", ctx=None, verbose=1, device=None):
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter, per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        n_atoms = self.D.shape[0]
        I_k = np.eye(n_atoms)

        parameters_config = dict(threshold=[], W_hessian=['sym'])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = {}
                if self.learn_th:
                    layer_params['threshold'] = np.ones(n_atoms) / self.L
                layer_params['W_hessian'] = I_k / self.L

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lmbd, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = np.zeros_like(np.array(x).dot(self.D.T)) if z0 is None else z0
        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(x, z0,
                                                            output_layer,
                                                            enable_none=False)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            th = layer_params['threshold']
            step_size = layer_params.get('step_size', 1.)
            W = self.D_.t().matmul(layer_params['W_hessian']) * step_size

            # apply one 'iteration'
            residual = z_hat.matmul(self.D_) - x
            z_hat = z_hat - residual.matmul(W)
            z_hat = soft_thresholding(z_hat, lmbd, th)

        return z_hat

    def _loss_fn(self, x, lmbd, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat).sum()
        return (loss + lmbd * reg) / n_samples


class StepLista(ListaBase):
    __doc__ = DOC_LISTA.format(type='L-ISTA step',
                               problem_name='LASSO',
                               descr='only learn a step size'
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer='one_shot', initial_parameters=[],
                 name="Step-LISTA", ctx=None, verbose=1, device=None):
        if not learn_th:
            print("With StepLista, 'learn_th' should be enable, 'learn_th' "
                  "switch to True")
            learn_th = True
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter,
                         per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        parameters_config = dict(step_size=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict(step_size=np.array(1 / self.L))

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lmbd, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = np.zeros_like(np.array(x).dot(self.D.T)) if z0 is None else z0
        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(x, z0,
                                                            output_layer,
                                                            enable_none=False)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            th = layer_params.get('step_size', 1/self.L)
            step_size = layer_params.get('step_size', 1.)
            W = self.D_.t() * step_size

            # apply one 'iteration'
            residual = z_hat.matmul(self.D_) - x
            z_hat = z_hat - residual.matmul(W)
            z_hat = soft_thresholding(z_hat, lmbd, th)

        return z_hat

    def _loss_fn(self, x, lmbd, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat).sum()
        return (loss + lmbd * reg) / n_samples


ALL_LISTA = dict(lista=Lista, alista=ALista, coupled=CoupledLista,
                 hessian=HessianLista, step=StepLista)


class StepSubGradLTV(ListaBase):
    __doc__ = DOC_LISTA.format(type='LTV step',
                               problem_name='TV',
                               descr='only learn a step size'
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer="one_shot", initial_parameters=[],
                 name="Step-LTV", ctx=None, verbose=1, device=None):
        if not learn_th:
            print("With StepSubGradLTV, 'learn_th' should be enable,"
                  "'learn_th' switch to True")
            learn_th = True
        super().__init__(D=D, n_layers=n_layers, learn_th=learn_th,
                         solver=solver, max_iter=max_iter,
                         per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        parameters_config = dict(step_size=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict(step_size=np.array(1.0e-8))  # XXX

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lmbd, z0=None, output_layer=None):
        """ Forward pass of the network. """
        x, z_hat, output_layer = self._check_forward_inputs(x, z0,
                                                            output_layer,
                                                            enable_none=True)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            step_size = layer_params.get('step_size', 1.)

            # apply one 'iteration'
            if z_hat is None:
                z_hat = step_size * x
            else:
                residual = z_hat - x
                reg = z_hat.matmul(self.D_).sign().matmul(self.D_.t())
                z_hat = z_hat - step_size * (residual + lmbd * reg)

        return z_hat

    def _loss_fn(self, x, lmbd, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat.matmul(self.D_)).sum()
        return (loss + lmbd * reg) / n_samples


ALL_LTV = dict(step=StepSubGradLTV)
