""" Base module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import importlib
import torch
import numpy as np
from .checks import check_tensor
from .analysis_loss_gradient import obj


AVAILABLE_CONTEXT = []  # a virer /!\

if importlib.util.find_spec('torch'):
    AVAILABLE_CONTEXT += ['torch']

if importlib.util.find_spec('tensorflow'):
    AVAILABLE_CONTEXT += ['tf']

assert len(AVAILABLE_CONTEXT) > 0, (
    "Should have at least one deep-learning framework in "
    "{'tensorflow' | 'pytorch'}"
)


DOC_LISTA = """ {type} network for the {problem_name} problem

    {descr}

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_dim)
        Dictionary for the considered sparse coding problem.
    n_layer : int
        Number of layers in the network.
    learn_th : bool (default: True)
        Wether to learn the thresholds or not.
    solver : str, (default: 'gradient_decent')
        Not implemented for now.
    max_iter : int (default: 100)
        Maximal number of iteration for the training of each layer.
    name : str (default: LISTA)
        Name of the model.
    verbose : int (default: 1)
        Verbosity level.
    device : str or None (default: None)
        Device on which the model is implemented. This parameter should be set
        according to the pytorch API (_eg_ 'cpu', 'gpu', 'gpu/1',..).
    """


class ListaBase(torch.nn.Module):

    __doc__ = DOC_LISTA.format(type='virtual-class', problem_name='LASSO',
                               descr='')

    def __init__(self, D, n_layers, learn_th=True, solver="gradient_descent",
                 max_iter=100, per_layer='recursive', initial_parameters=[],
                 name="LISTA", ctx=None, verbose=1, device=None):
        if ctx:
            msg = "Context {} is not available on this computer."
            assert ctx in AVAILABLE_CONTEXT, msg.format(ctx)
        else:
            ctx = AVAILABLE_CONTEXT[0]

        self.name = name
        self._ctx = ctx
        self.device = device
        self.verbose = verbose

        self.solver = solver
        self.max_iter = max_iter
        self.per_layer = per_layer

        self.n_layers = n_layers

        self.learn_th = learn_th
        self.pre_gradient_hooks = {"sym": []}

        self.D = np.array(D)
        self.D_ = check_tensor(self.D, device=device)
        self.B = D.dot(D.T)
        self.L = np.linalg.norm(self.B, ord=2)

        self.layers_parameters = []

        super().__init__()

        self._init_network_parameters(initial_parameters=initial_parameters)

    def export_parameters(self):
        """ Return a list with all the parameters of the network.

        This list can be used to init a new network which will have the same
        output. Usefull to save the parameters.
        """
        return [
            {k: p.detach().cpu().numpy() for k, p in layer_parameters.items()}
            for layer_parameters in self.layers_parameters
        ]

    def get_parameters(self, name):
        """ Return a list with the parameter name of each layer in the network.
        """
        return [
            layer_parameters[name].detach().cpu().numpy()
            if name in layer_parameters else None
            for layer_parameters in self.layers_parameters
        ]

    def set_parameters(self, name, values, offset=None):
        """ Return a list with the parameter name of each layer in the network.
        """
        layers_parameters = self.layers_parameters[offset:]
        if type(values) != list:
            values = [values] * len(layers_parameters)
        for layer_parameters, value in zip(layers_parameters, values):
            if name in layer_parameters:
                layer_parameters[name].data = check_tensor(value)

    def fit(self, x, lbda):
        """ Compute the output of the network, given x and regularization lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        """
        x = check_tensor(x, device=self.device)

        if self.solver == "gradient_descent":
            self._fit_all_network_batch_gradient_descent(x, lbda)
        else:
            raise NotImplementedError(f"'solver' parameter "  # noqa: E999
                                      f"should be in ['gradient_descent']"
                                      f", got {self.solver}")
        return self

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

    def score(self, x, lbda, z0=None, output_layer=None):
        """ Compute the loss for the network's output

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
            layers of the network. Ifs set to None, output the network's last
            layer.
        """
        x = check_tensor(x, device=self.device)
        with torch.no_grad():
            return self._loss_fn(x, lbda, self(x, lbda, z0=z0,
                                 output_layer=output_layer)).cpu().numpy()

    def compute_loss(self, x, lbda, z0=None):
        """ Compute the loss for the network's output at each layer

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        z0 : ndarray, shape (n_samples, n_atoms) (default: None)
            Initial point for the optimization algorithm. If None, the
            algorithm starts from 0
        """
        x = check_tensor(x, device=self.device)
        loss = []
        with torch.no_grad():
            for output_layer in range(self.n_layers):
                loss.append(self._loss_fn(
                    x, lbda,
                    self(x, lbda, z0=z0, output_layer=output_layer + 1)
                    ).cpu().numpy())
        return np.array(loss)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        raise NotImplementedError('ListaBase is a virtual class and should '
                                  'not be instanciate')

    def _fit_all_network_batch_gradient_descent(self, x, lbda):
        """ Fit the parameters of the network. """
        if self.per_layer == 'one_shot':
            params = [p for layer_parameters in self.layers_parameters
                      for p in layer_parameters.values()]
            self._fit_sub_net_batch_gd(x, lbda, params, self.n_layers,
                                       self.max_iter)

        elif self.per_layer == 'recursive':
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(0, self.max_iter, self.n_layers+1,
                                            dtype=int))
            for n_layer, max_iter in zip(layers, max_iters):
                params = [p for lp in self.layers_parameters
                          for p in lp.values()]
                self._fit_sub_net_batch_gd(x, lbda, params, n_layer, max_iter)

        elif self.per_layer == 'greedy':
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(0, self.max_iter, self.n_layers+1,
                                            dtype=int))
            for n_layer, max_iter in zip(layers, max_iters):
                params = [p for lp in self.layers_parameters[:n_layer]
                          for p in lp.values()]
                self._fit_sub_net_batch_gd(x, lbda, params, n_layer, max_iter)

        else:
            raise ValueError(f"per_layer should belong to ['recursive', "
                             f"'one_shot', 'greedy'], got {self.per_layer}")

        if self.verbose:
            print(f"\r[{self.name}-{self.n_layers}] Fitting model: done"
                  .ljust(80))

        return self

    def _fit_sub_net_batch_gd(self, x, lbda, parameters, n_layer, max_iter,
                              max_iter_line_search=10, eps=1.0e-20):
        """ Fit the parameters of the sub-network. """
        with torch.no_grad():
            z_hat = self(x, lbda, output_layer=n_layer)
            self.training_loss_ = [float(self._loss_fn(x, lbda, z_hat))]
        self.norm_grad_ = []

        verbose_rate = 50
        lr = 1.0
        is_converged = False

        for i in range(max_iter):

            # Verbosity
            if self.verbose > 1 and i % verbose_rate == 0:
                print(f"\r[{self.name} - layer{n_layer}] "  # noqa: E999
                      f"Fitting, step_size={lr:.2e}, "
                      f"iter={i}/{max_iter}, "
                      f"loss={self.training_loss_[-1]:.3e}")

            # Gradient computation
            self._update_gradient(x, lbda, n_layer)

            # Back-tracking line search descent step
            for _ in range(max_iter_line_search): 

                # Next gradient step
                max_norm_grad = self._update_parameters(parameters, lr=lr)

                # Compute new possible loss
                with torch.no_grad():
                    z_hat = self(x, lbda, output_layer=n_layer)
                    loss_value = float(self._loss_fn(x, lbda, z_hat))

                # accepting the point
                if self.training_loss_[-1] > loss_value:
                    self.training_loss_.append(loss_value)
                    self.norm_grad_.append(max_norm_grad)
                    lr *= 2**(max_iter_line_search / 4)
                    break
                # rejecting the point
                else:
                    _ = self._update_parameters(parameters, lr=-lr)
                    lr /= 2.0

            # Stopping criterion
            if lr < 1e-20:
                is_converged = True
            if len(self.training_loss_) > 1:
                if self.training_loss_[-2] - self.training_loss_[-1] < eps:
                    is_converged = True

            if is_converged:
                if self.verbose:
                    print(f"\r[{self.name} - layer{n_layer}] "
                        f"Early-stop, step_size={lr:.2e}, ",
                        f"iter={i}/{max_iter}, "
                        f"final-loss={self.training_loss_[-1]:.3e}")
                break

        assert np.all(np.diff(self.training_loss_) < 0.0), "During training loss function has increased!"

    def _update_gradient(self, x, lbda, n_layer):
        """ Gradient update. """
        # init gradient
        self.zero_grad()
        # init back-propagation
        z_hat = self(x, lbda, output_layer=n_layer)
        loss = self._loss_fn(x, lbda, z_hat)
        loss.backward()

    def _update_parameters(self, parameters, lr):
        """ Parameters update step for the gradient descent. """
        self._saved_gradient = []
        max_norm_grad = 0.0

        for param in parameters:
            if param.grad is not None:
                # do a descent step
                param.data.add_(-lr, param.grad.data)
                # compute gradient max norm
                current_norm_grad = param.grad.data.detach().abs().max()
                max_norm_grad = np.maximum(max_norm_grad, current_norm_grad)
                # save gradient
                self._saved_gradient.append(param.grad.data.clone())
            else:
                # save gradient
                self._saved_gradient.append(None)

        return float(max_norm_grad)

    def _check_forward_inputs(self, x, z0, output_layer, enable_none=False):
        """ Format properly the inputs for the 'forward' method. """
        x_none_ok = (x is None) and enable_none
        x = x if x_none_ok else check_tensor(x, device=self.device)
        z0_none_ok = (z0 is None) and enable_none
        z0 = z0 if z0_none_ok else check_tensor(z0, device=self.device)
        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError(f"Requested output from out-of-bound layer "
                             f"output_layer={output_layer} "
                             f"(n_layers={self.n_layers})")
        return x, z0, output_layer

    def _tensorized_and_hooked_parameters(self, layer, layer_params,
                                          parameters_config):
        """ Transform all the parameters to learnable Tensor and store
        parameters hooks and register them."""
        layer_params = {
            k: torch.nn.Parameter(check_tensor(p, device=self.device))
            for k, p in layer_params.items()}
        for name, p in layer_params.items():
            self.register_parameter("layer{}-{}".format(layer, name), p)
            for h in parameters_config[name]:
                self.pre_gradient_hooks[h].append(p)
        return layer_params
