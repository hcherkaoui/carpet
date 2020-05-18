""" Base module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import torch
import numpy as np
from .checks import check_tensor
from .checks import check_parameter
from .parameters import list_parameters_from_groups


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

        # specific initialization networks
        self.parameter_groups = {}
        self.force_learn_groups = []
        super().__init__()

        # inti network
        self._init_network_parameters(initial_parameters=initial_parameters)

    def _init_network_parameters(self, initial_parameters=None):
        """ Initialize the parameters of the network. """
        if initial_parameters is None:
            initial_parameters = {}

        self.get_global_parameters(initial_parameters)

        for layer_id in range(self.n_layers):
            group_name = f'layer-{layer_id}'
            if group_name in initial_parameters.keys():
                layer_params = initial_parameters[group_name]
            else:
                layer_params = self.get_initial_layer_parameters(layer_id)

            layer_params = self._register_parameters(layer_params,
                                                     group_name=group_name)

    def export_parameters(self):
        """ Return a list with all the parameters of the network.

        This list can be used to init a new network which will have the same
        output. Usefull to save the parameters.
        """
        return {
            group_name: {k: p.detach().cpu().numpy()
                         for k, p in group_params.items()}
            for group_name, group_params in self.parameter_groups.items()
        }

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
        lbda = check_tensor(lbda, device=self.device)
        self._fit_all_network_batch_gradient_descent(x, lbda)
        return self

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
        x = check_tensor(x, device=self.device)
        lbda = check_tensor(lbda, device=self.device)
        with torch.no_grad():
            return self(
                x, lbda, output_layer=output_layer
            ).detach().cpu().numpy()

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
        lbda = check_tensor(lbda, device=self.device)
        with torch.no_grad():
            return self._loss_fn(
                x, lbda, self(x, lbda, output_layer=output_layer)
            ).detach().cpu().numpy()

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

    def get_initial_layer_parameters(self, layer):
        """ Initialize the parameters of one layer of the network. """
        raise NotImplementedError('ListaBase is a virtual class and should '
                                  'not be instanciated')

    def get_global_parameters(self, initial_parameters):
        """ Initialize the global parameters of the network. """
        pass

    def _fit_all_network_batch_gradient_descent(self, x, lbda):
        """ Fit the parameters of the network. """
        if self.net_solver_type == 'one_shot':
            params = list(self.parameters())  # Get all parameters
            for group in self.force_learn_groups:
                assert len(set(self.parameter_groups[group].values())
                           - set(params)) == 0
            self._fit_sub_net_batch_gd(x, lbda, params, self.n_layers,
                                       self.max_iter)

        elif self.net_solver_type == 'recursive':
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(0, self.max_iter, self.n_layers+1,
                                            dtype=int))
            for layer_id, max_iter in zip(layers, max_iters):
                params = list(self.parameters())  # Get all parameters
                for group in self.force_learn_groups:
                    assert len(set(self.parameter_groups[group].values())
                               - set(params)) == 0
                self._fit_sub_net_batch_gd(x, lbda, params, layer_id, max_iter)

        elif self.net_solver_type == 'greedy':
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(0, self.max_iter, self.n_layers+1,
                                            dtype=int))
            for layer_id, max_iter in zip(layers, max_iters):
                group_layers = [f'layer-{lid}' for lid in range(layer_id)]
                params = list_parameters_from_groups(
                    self.parameter_groups,
                    self.force_learn_groups + group_layers)
                self._fit_sub_net_batch_gd(x, lbda, params, layer_id, max_iter)

        else:
            raise ValueError(f"net_solver_type should belong to "
                             f"['recursive', 'one_shot', 'greedy']"
                             f", got {self.net_solver_type}")

        if self.verbose:
            print(f"\r[{self.name}-{self.n_layers}] Fitting model: done"
                  .ljust(80))

        return self

    def _fit_sub_net_batch_gd(self, x, lbda, params, layer_id, max_iter,
                              output_layer=None, max_iter_line_search=100,
                              eps=1.0e-20):
        """ Fit the parameters of the sub-network. """
        if output_layer is None:
            output_layer = layer_id
        with torch.no_grad():
            z = self(x, lbda, output_layer=output_layer)
            prev_loss = self._loss_fn(x, lbda, z)
            self.training_loss_ = [float(prev_loss)]
        self.norm_grad_ = []

        verbose_rate = 50
        lr = 1.0
        is_converged = False

        i = 0

        if self.verbose > 1:
            print(f"\r[{self.name} - layer{layer_id}] "  # noqa: E999
                  f"Fitting, step_size={lr:.2e}, "
                  f"iter={i}/{max_iter}, "
                  f"loss={self.training_loss_[-1]:.3e}")

        while i < max_iter:

            i += 1

            # Verbosity
            if self.verbose > 1 and i % verbose_rate == 0:
                print(f"\r[{self.name} - layer{layer_id}] "  # noqa: E999
                      f"Fitting, step_size={lr:.2e}, "
                      f"iter={i}/{max_iter}, "
                      f"loss={self.training_loss_[-1]:.3e}")
            if self.verbose > 0:
                p = (layer_id - 1 + i/max_iter) / self.n_layers
                print(f"\rTraining... {p:7.2%}", end='', flush=True)

            # Gradient computation
            self._compute_gradient(x, lbda, output_layer)

            # Back-tracking line search descent step
            for _ in range(max_iter_line_search):

                # Next gradient step
                max_norm_grad = self._update_parameters(params, lr=lr)

                # Compute new possible loss
                with torch.no_grad():
                    z = self(x, lbda, output_layer=output_layer)
                    loss_value = self._loss_fn(x, lbda, z)

                # accepting the point
                if prev_loss > loss_value:
                    prev_loss = loss_value
                    self.training_loss_.append(float(loss_value))
                    self.norm_grad_.append(float(max_norm_grad))
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
            print(f"\r[{self.name} - layer{layer_id}] Finished "
                  f"({converging_status}converged), step_size={lr:.2e}, "
                  f"iter={i}/{max_iter}, "
                  f"final-loss={self.training_loss_[-1]:.6e}")

        msg = "Loss function has increased during training!"
        assert np.all(np.diff(self.training_loss_) < 0.0), msg

    def _compute_gradient(self, x, lbda, layer_id):
        """ Gradient update. """
        # init gradient
        self.zero_grad()
        # init back-propagation
        z = self(x, lbda, output_layer=layer_id)
        loss = self._loss_fn(x, lbda, z)
        # Compute gradient
        loss.backward()

    def _update_parameters(self, parameters, lr):
        """ Parameters update step for the gradient descent. """
        max_norm_grad = 0.0

        for param in parameters:
            if param.grad is not None:
                # do a descent step
                param.data.add_(-lr, param.grad.data)
                # compute gradient max norm
                current_norm_grad = param.grad.data.abs().max()
                max_norm_grad = max(max_norm_grad, current_norm_grad)

        return max_norm_grad

    def _register_parameters(self, group_params, group_name):
        """ Transform all the parameters to learnable Tensor"""
        # transform all pre-parameters into learnable torch.nn.Parameters
        group_params = {
            k: check_parameter(p, device=self.device)
            for k, p in group_params.items()
        }
        self.parameter_groups[group_name] = group_params
        for name, p in group_params.items():
            param_key = f"{group_name}:{name}"
            self.register_parameter(param_key, p)

        return group_params

    def check_output_layer(self, output_layer):
        if output_layer is None:
            output_layer = self.n_layers
        if output_layer > self.n_layers:
            raise ValueError(f"Requested output from out-of-bound layer "
                             f"output_layer={output_layer} "
                             f"(n_layers={self.n_layers})")
        return output_layer
