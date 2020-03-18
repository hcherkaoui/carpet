import importlib
import torch
import numpy as np
from .checks import check_tensor
from .synthesis_loss_gradient import obj


AVAILABLE_CONTEXT = []

if importlib.util.find_spec('torch'):
    AVAILABLE_CONTEXT += ['torch']

if importlib.util.find_spec('tensorflow'):
    AVAILABLE_CONTEXT += ['tf']

assert len(AVAILABLE_CONTEXT) > 0, (
    "Should have at least one deep-learning framework in "
    "{'tensorflow' | 'pytorch'}"
)


def symmetric_gradient(p):
    """ Constrain the gradient to be symmetric. """
    p.grad.data.set_(p.grad.data + p.grad.data.t())


GRADIENT_HOOKS = {
    'sym': symmetric_gradient,
}


class ListaBase(torch.nn.Module):
    """ L-ISTA network for the LASSO problem

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_dim)
        Dictionary for the considered sparse coding problem.
    n_layer : int
        Number of layers in the network.
    parametrization : str, (default: "coupled")
        Parametrization for the weight of the network. Should be one of:
        - 'lista': original parametrization from Gregor and Le Cun (2010).
        - 'coupled': one weight parametrization from Chen et al (2018).
        - 'alista': analytic weights from Chen et al (2019).
        - 'hessian': one weight parametrization as a quasi newton technique.
        - 'step': only learn a step size
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
    def __init__(self, D, n_layers, parametrization="coupled", learn_th=True,
                 solver="gradient_descent", max_iter=100, per_layer='auto',
                 initial_parameters=[], name="LISTA", ctx=None, verbose=1,
                 device=None):
        if ctx:
            msg = "Context {} is not available on this computer."
            assert ctx in AVAILABLE_CONTEXT, msg.format(ctx)
        else:
            ctx = AVAILABLE_CONTEXT[0]

        if parametrization in ['step', 'coupled_step'] and not learn_th:
            raise ValueError("It is not possible to use parametrization "
                             "with step and learn_th=False")

        if per_layer == 'auto':
            if parametrization == 'step':
                per_layer = 'oneshot'
            else:
                per_layer = "recursive"

        self.name = name
        self._ctx = ctx
        self.device = device
        self.verbose = verbose

        self.solver = solver
        self.max_iter = max_iter
        self.per_layer = per_layer

        self.n_layers = n_layers

        self.learn_th = learn_th
        self.parametrization = parametrization
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

    def fit(self, x, lmbd):
        """ Compute the output of the network, given x and regularization lmbd

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lmbd: float
            Regularization level for the optimization problem.
        """
        # Compat numpy
        x = check_tensor(x, device=self.device)

        if self.solver == "gradient_descent":
            self._fit_batch_gradient_descent(x, lmbd)
        else:
            raise NotImplementedError("'solver' parameter should be in "
                                      "{'gradient_descent'}")
        return self

    def transform(self, x, lmbd, z0=None, output_layer=None):
        """ Compute the output of the network, given x and regularization lmbd

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lmbd: float
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
            return self(x, lmbd, z0=z0,
                        output_layer=output_layer).cpu().numpy()

    def score(self, x, lmbd, z0=None, output_layer=None):
        """ Compute the loss for the network's output

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lmbd: float
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
            return self._loss_fn(x, lmbd, self(x, lmbd, z0=z0,
                                 output_layer=output_layer)).cpu().numpy()

    def compute_loss(self, x, lmbd, z0=None):
        """ Compute the loss for the network's output at each layer

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lmbd: float
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
                    x, lmbd,
                    self(x, lmbd, z0=z0, output_layer=output_layer + 1)
                    ).cpu().numpy())
        return np.array(loss)

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        raise NotImplementedError()

    def _fit_batch_gradient_descent(self, x, lmbd):

        if self.verbose > 1:
            # compute fix point
            z_hat = self.transform(x, lmbd)
            for i in range(100):
                z_hat = self.transform(x, lmbd, z0=z_hat)
            c_star = obj_tensor(z_hat.cpu().numpy(), self.D, x, lmbd)

        parameters = [p for layer_parameters in self.layers_parameters
                      for p in layer_parameters.values()]

        training_loss = []
        norm_gradients = []
        if self.per_layer == 'oneshot':
            layers = [self.n_layers]
            max_iters = [self.max_iter]
        else:
            layers = range(1, self.n_layers + 1)
            max_iters = np.diff(np.linspace(
                0, self.max_iter, self.n_layers + 1, dtype=int))

        for n_layer, max_iter in zip(layers, max_iters):
            lr = 1

            if self.per_layer == "greedy":
                parameters = [
                    p for lp in self.layers_parameters[:n_layer]
                    for p in lp.values()
                ]
            else:
                parameters = [
                    p for lp in self.layers_parameters for p in lp.values()
                ]
            i = 0
            while i < max_iter:

                # Compute the forward operator
                self.zero_grad()
                if self.per_layer == "recursive":
                    z_hat = self(x, lmbd, output_layer=n_layer)
                else:
                    z_hat = self(x, lmbd)
                loss = self._loss_fn(x, lmbd, z_hat)

                # Verbosity of the output
                if self.verbose > 5 and i % 10 == 0:
                    loss_val = loss.detach().cpu().numpy()
                    print(i, loss_val - c_star)
                elif self.verbose > 0 and i % 50 == 0:
                    print(f"\rFitting model (layer {n_layer}/{self.n_layers})"
                          f" : {(i+1) / max_iter:7.2%}", end="", flush=True)

                # Back-tracking line search
                if len(training_loss) > 0 and training_loss[-1] < float(loss):
                    lr = self._backtrack_parameters(parameters, lr)
                    if lr < 1e-20:
                        if self.verbose:
                            print(f"\r[{self.name} - layer{n_layer}] "
                                  f"Converged, step_size={lr:.2e}, "
                                  f"norm_g={norm_gradients[-1]:.2e}")
                        break
                    continue

                # Accepting the previous point
                training_loss.append(float(loss))
                i += 1

                # Next gradient iterate
                loss.backward()
                lr, norm_g = self._update_parameters(parameters, lr=lr)
                norm_gradients.append(norm_g)

        self.training_loss_ = training_loss
        self.norm_gradients_ = norm_gradients
        if self.verbose:
            print(f"\r[{self.name}-{self.n_layers}] Fitting model: done"
                  .ljust(80))
        return self

    def _loss_fn(self, x, lmbd, z_hat):
        x = check_tensor(x, device=self.device)
        res = z_hat.matmul(self.D_) - x
        return 0.5 * (res * res).sum() + lmbd * torch.abs(z_hat).sum()

    def _update_parameters(self, parameters, lr):
        lr = min(4 * lr, 1e12)

        self._saved_gradient = []

        for hook, list_params in self.pre_gradient_hooks.items():
            for p in list_params:
                if p.grad is not None:
                    GRADIENT_HOOKS[hook](p)

        norm_g = 0
        for i, p in enumerate(parameters):
            if p.grad is None:
                self._saved_gradient.append(None)
                continue

            p.data.add_(-lr, p.grad.data)
            self._saved_gradient.append(p.grad.data.clone())
            norm_g = max(norm_g, p.grad.data.detach().abs().max())

        return lr, float(norm_g)

    def _backtrack_parameters(self, parameters, lr):
        lr /= 2
        for p, g in zip(parameters, self._saved_gradient):
            if g is None:
                continue
            p.data.add_(lr, g)
        return lr
