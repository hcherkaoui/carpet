""" Base module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .checks import check_tensor


DOC_LISTA = """ {type} network for the {problem_name} problem

    {descr}

    Parameters
    ----------
    A : ndarray, shape (n_atoms, n_dim)
        Dictionary for the considered sparse coding problem.
    D : ndarray, shape (n_dim, n_dim-1)
        Integration or differentiation operator
    n_layers : int
        Number of layers in the network.
    learn_th : bool (default: True)
        Wether to learn the thresholds or not.
    net_solver_type : str, (default: 'gradient_decent')
        Not implemented for now.
    initial_parameters :  list of dict,
        Layer-wise initial parameters
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

    __doc__ = DOC_LISTA.format(type='virtual-class', problem_name='',
                               descr='')

    def __init__(self, n_layers, learn_th=True, max_iter=100,
                 net_solver_type='one_shot', initial_parameters=None,
                 name="LISTA", verbose=0, device=None):
        # general parameters
        self.name = name
        self.device = device
        self.verbose = verbose

        # network training solver parameters
        self.max_iter = max_iter
        self.net_solver_type = net_solver_type

        # networks meta-parameters
        self.n_layers = n_layers
        self.learn_th = learn_th

        # psecific initialization networks
        self.layers_parameters = []
        super().__init__()

        # inti network
        self._init_network_parameters(initial_parameters=initial_parameters)

    def _init_network_parameters(self, initial_parameters=None):
        """ Initialize the parameters of the network. """
        if initial_parameters is None:
            initial_parameters = []

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                layer_params = self.get_layer_parameters(layer)

            layer_params = self._tensorized_parameters(layer, layer_params)
            self.layers_parameters += [layer_params]

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
                layer_parameters[name].data = check_tensor(value,
                                                           device=self.device)

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
        lbda = float(lbda)
        self._fit_all_network_batch_gradient_descent(x, lbda)
        return self

    def score(self, x, lbda, output_layer=None):
        """ Compute the loss for the network's output

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. Ifs set to None, output the network's last
            layer.
        """
        x = check_tensor(x, device=self.device)
        with torch.no_grad():
            return self._loss_fn(x, lbda, self(x, lbda,
                                 output_layer=output_layer)).cpu().numpy()

    def compute_loss(self, x, lbda):
        """ Compute the loss for the network's output at each layer

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        """
        x = check_tensor(x, device=self.device)
        loss = []
        with torch.no_grad():
            for output_layer in range(self.n_layers):
                z = self(x, lbda, output_layer=output_layer + 1)
                loss.append(self._loss_fn(x, lbda, z).cpu().numpy())
        return np.array(loss)

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
        raise NotImplementedError('ListaBase is a virtual class and should '
                                  'not be instanciate')

    def get_layer_parameters(self, layer):
        """ Initialize the parameters of one layer of the network. """
        raise NotImplementedError('ListaBase is a virtual class and should '
                                  'not be instanciated')

    def _fit_all_network_batch_gradient_descent(self, x, lbda):
        """ Fit the parameters of the network. """
        if self.net_solver_type == 'one_shot':
            params = list(self.parameters())
            self._fit_sub_net_batch_gd(x, lbda, params, self.n_layers,
                                       self.max_iter)

        elif self.net_solver_type == 'recursive':
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(0, self.max_iter, self.n_layers+1,
                                            dtype=int))
            for id_layer, max_iter in zip(layers, max_iters):
                params = [p for lp in self.layers_parameters
                          for p in lp.values()]
                self._fit_sub_net_batch_gd(x, lbda, params, id_layer, max_iter)

        elif self.net_solver_type == 'greedy':
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(0, self.max_iter, self.n_layers+1,
                                            dtype=int))
            for id_layer, max_iter in zip(layers, max_iters):
                params = [p for lp in self.layers_parameters[:id_layer]
                          for p in lp.values()]
                self._fit_sub_net_batch_gd(x, lbda, params, id_layer, max_iter)

        else:
            raise ValueError(f"net_solver_type should belong to "
                             f"['recursive', 'one_shot', 'greedy']"
                             f", got {self.net_solver_type}")

        if self.verbose:
            print(f"\r[{self.name}-{self.n_layers}] Fitting model: done"
                  .ljust(80))

        return self

    def _fit_sub_net_batch_gd(self, x, lbda, params, id_layer, max_iter,
                              max_iter_line_search=10, eps=1.0e-20):
        """ Fit the parameters of the sub-network. """
        with torch.no_grad():
            z = self(x, lbda, output_layer=id_layer)
            self.training_loss_ = [float(self._loss_fn(x, lbda, z))]
        self.norm_grad_ = []

        verbose_rate = 50
        lr = 1.0
        is_converged = False

        i = 0

        if self.verbose > 1:
            print(f"\r[{self.name} - layer{id_layer}] "  # noqa: E999
                  f"Fitting, step_size={lr:.2e}, "
                  f"iter={i}/{max_iter}, "
                  f"loss={self.training_loss_[-1]:.3e}")

        while i < max_iter:

            i += 1

            # Verbosity
            if self.verbose > 1 and i % verbose_rate == 0:
                print(f"\r[{self.name} - layer{id_layer}] "  # noqa: E999
                      f"Fitting, step_size={lr:.2e}, "
                      f"iter={i}/{max_iter}, "
                      f"loss={self.training_loss_[-1]:.3e}")
            else:
                p = (id_layer - 1 + i/max_iter) / self.n_layers
                print(f"\rTraining... {p:7.2%}", end='', flush=True)

            # Gradient computation
            self._update_gradient(x, lbda, id_layer)

            # Back-tracking line search descent step
            for _ in range(max_iter_line_search):

                # Next gradient step
                max_norm_grad = self._update_parameters(params, lr=lr)

                # Compute new possible loss
                with torch.no_grad():
                    z = self(x, lbda, output_layer=id_layer)
                    loss_value = float(self._loss_fn(x, lbda, z))

                # accepting the point
                if self.training_loss_[-1] > loss_value:
                    self.training_loss_.append(loss_value)
                    self.norm_grad_.append(max_norm_grad)
                    lr *= 2**(max_iter_line_search / 4)
                    break
                # rejecting the point
                else:
                    _ = self._update_parameters(params, lr=-lr)  # noqa: F841
                    lr /= 2.0

            # Stopping criterion
            if lr < 1e-20:
                is_converged = True
            if len(self.training_loss_) > 1:
                if self.training_loss_[-2] - self.training_loss_[-1] < eps:
                    is_converged = True

            # Early stopping
            if is_converged:
                break

        if self.verbose:
            converging_status = '' if is_converged else 'not '
            print(f"\r[{self.name} - layer{id_layer}] Finished "
                  f"({converging_status}converged), step_size={lr:.2e}, "
                  f"iter={i}/{max_iter}, "
                  f"final-loss={self.training_loss_[-1]:.6e}")

        msg = "Loss function has increased during training!"
        assert np.all(np.diff(self.training_loss_) < 0.0), msg

    def _update_gradient(self, x, lbda, id_layer):
        """ Gradient update. """
        # init gradient
        self.zero_grad()
        # init back-propagation
        z = self(x, lbda, output_layer=id_layer)
        loss = self._loss_fn(x, lbda, z)
        # Compute gradient
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
                max_norm_grad = max(max_norm_grad, float(current_norm_grad))
                # save gradient
                self._saved_gradient.append(param.grad.data.clone())
            else:
                # save gradient
                self._saved_gradient.append(None)

        return float(max_norm_grad)

    def _check_forward_inputs(self, x, output_layer, enable_none=False):
        """ Format properly the inputs for the 'forward' method. """
        x_none_ok = (x is None) and enable_none
        x = x if x_none_ok else check_tensor(x, device=self.device)
        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError(f"Requested output from out-of-bound layer "
                             f"output_layer={output_layer} "
                             f"(n_layers={self.n_layers})")
        return x, output_layer

    def _tensorized_parameters(self, layer, layer_params):
        """ Transform all the parameters to learnable Tensor"""
        layer_params = {
            k: torch.nn.Parameter(check_tensor(p, device=self.device))
            for k, p in layer_params.items()}
        for name, p in layer_params.items():
            self.register_parameter("layer{}-{}".format(layer, name), p)
        return layer_params
