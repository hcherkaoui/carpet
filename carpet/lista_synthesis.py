""" Module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .lista_base import ListaBase, DOC_LISTA
from .proximity import pseudo_soft_th_tensor
from .utils import get_alista_weights
from .checks import check_tensor


class ListaLASSO(ListaBase):
    __doc__ = DOC_LISTA.format(
                type='learned-ISTA original',
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
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0 / self.L)
                layer_params['Wz'] = I_k - self.B / self.L
                layer_params['Wx'] = self.D.T / self.L

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = x

        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(
                                        x, z0, output_layer, enable_none=True)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0)
            Wx = layer_params['Wx']
            Wz = layer_params['Wz']

            # apply one 'iteration'
            z_hat = (z_hat.matmul(Wz) + x.matmul(Wx))
            z_hat = pseudo_soft_th_tensor(z_hat, lbda, mul_lbda)

        return z_hat

    def transform(self, x, lbda, z0=None, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        z0 : ndarray, shape (n_samples, n_atoms) (default: None)
            Initial point for the optimization algorithm. If None, the
            algorithm starts from 0
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            return self(x, lbda, z0=z0,
                        output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat[:, 1:]).sum()
        return (loss + lbda * reg) / n_samples


class CoupledIstaLASSO(ListaBase):
    __doc__ = DOC_LISTA.format(
                    type='learned-ISTA coupled',
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
        parameters_config = dict(threshold=[], W_coupled=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0 / self.L)
                layer_params['W_coupled'] = self.D.T / self.L

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = x

        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(
                                        x, z0, output_layer, enable_none=True)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0)
            W = layer_params['W_coupled']

            # apply one 'iteration'
            residual = z_hat.matmul(self.D_) - x
            z_hat = z_hat - residual.matmul(W)
            z_hat = pseudo_soft_th_tensor(z_hat, lbda, mul_lbda)

        return z_hat

    def transform(self, x, lbda, z0=None, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        z0 : ndarray, shape (n_samples, n_atoms) (default: None)
            Initial point for the optimization algorithm. If None, the
            algorithm starts from 0
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            return self(x, lbda, z0=z0,
                        output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat[:, 1:]).sum()
        return (loss + lbda * reg) / n_samples


class AIstaLASSO(ListaBase):
    __doc__ = DOC_LISTA.format(
                            type='learned-ISTA alista',
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

        parameters_config = dict(threshold=[], step_size=[])

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = dict()
                if self.learn_th:
                    layer_params['threshold'] = np.array(1.0 / self.L)
                layer_params['step_size'] = np.array(1.0 / self.L)

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = x

        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(
                                        x, z0, output_layer, enable_none=True)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0)
            step_size = layer_params.get('step_size', 1.0)
            W = self.W * step_size

            # apply one 'iteration'
            residual = z_hat.matmul(self.D_) - x
            z_hat = z_hat - residual.matmul(W)
            z_hat = pseudo_soft_th_tensor(z_hat, lbda, mul_lbda)

        return z_hat

    def transform(self, x, lbda, z0=None, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        z0 : ndarray, shape (n_samples, n_atoms) (default: None)
            Initial point for the optimization algorithm. If None, the
            algorithm starts from 0
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            return self(x, lbda, z0=z0,
                        output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat[:, 1:]).sum()
        return (loss + lbda * reg) / n_samples


class StepIstaLASSO(ListaBase):
    __doc__ = DOC_LISTA.format(type='learned-ISTA step',
                               problem_name='LASSO',
                               descr='only learn a step size'
                               )

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer='one_shot', initial_parameters=[],
                 name="Step-LISTA", ctx=None, verbose=1, device=None):
        if learn_th:
            print("In 'StepIstaLASSO' 'learn_th' can't be enable, switch it "
                  "to False")
            learn_th = True
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
                    layer_params['threshold'] = np.array(1.0 / self.L)
                layer_params['step_size'] = np.array(1.0 / self.L)

            layer_params = self._tensorized_and_hooked_parameters(
                                        layer, layer_params, parameters_config)
            self.layers_parameters += [layer_params]

    def forward(self, x, lbda, z0=None, output_layer=None):
        """ Forward pass of the network. """
        # initialize the descent
        z0 = x

        # check inputs
        x, z_hat, output_layer = self._check_forward_inputs(
                                        x, z0, output_layer, enable_none=True)

        for layer_params in self.layers_parameters[:output_layer]:
            # retrieve parameters
            mul_lbda = layer_params.get('threshold', 1.0)
            step_size = layer_params.get('step_size', 1.0)
            W = self.D_.t() * step_size

            # apply one 'iteration'
            residual = z_hat.matmul(self.D_) - x
            z_hat = z_hat - residual.matmul(W)
            z_hat = pseudo_soft_th_tensor(z_hat, lbda, mul_lbda)

        return z_hat

    def transform(self, x, lbda, z0=None, output_layer=None):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        z0 : ndarray, shape (n_samples, n_atoms) (default: None)
            Initial point for the optimization algorithm. If None, the
            algorithm starts from 0
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network.
        """
        with torch.no_grad():
            return self(x, lbda, z0=z0,
                        output_layer=output_layer).cpu().numpy()

    def _loss_fn(self, x, lbda, z_hat):
        """ Target loss function. """
        n_samples = x.shape[0]
        x = check_tensor(x, device=self.device)
        z_hat = check_tensor(z_hat, device=self.device)
        residual = z_hat.matmul(self.D_) - x
        loss = 0.5 * (residual * residual).sum()
        reg = torch.abs(z_hat[:, 1:]).sum()
        return (loss + lbda * reg) / n_samples


ALL_LISTA = dict(lista=ListaLASSO, alista=AIstaLASSO, coupled=CoupledIstaLASSO,
                 step=StepIstaLASSO)
