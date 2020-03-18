import torch
import numpy as np
from .lista_base import ListaBase
from .proximity import soft_thresholding
from .utils import get_alista_weights
from .checks import check_tensor


PARAMETRIZATIONS = {
    "lista": {
        'threshold': [],
        'Wx': [],
        'Wz': [],
    },
    "hessian": {
        'threshold': [],
        'W_hessian': ["sym"],
    },
    "coupled": {
        'threshold': [],
        'W_coupled': [],
    },
    "alista": {
        'threshold': [],
        'step_size': [],
    },
    "step": {
        'step_size': [],
    },
    "first_step": {
        'step_size': [],
        'threshold': [],
        'W_coupled': [],
    },
}


class Lista(ListaBase):
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
        super().__init__(D=D, n_layers=n_layers,
                         parametrization=parametrization, learn_th=learn_th,
                         solver=solver, max_iter=max_iter, per_layer=per_layer,
                         initial_parameters=initial_parameters, name=name,
                         ctx=ctx, verbose=verbose, device=device)

        if parametrization not in PARAMETRIZATIONS:
            raise NotImplementedError("Could not find parametrization='{}'. "
                                      "Should be in {}".format(
                                          parametrization, PARAMETRIZATIONS
                                      ))

    def forward(self, x, lmbd, z0=None, output_layer=None):
        # Compat numpy
        x = check_tensor(x, device=self.device)
        z0 = check_tensor(z0, device=self.device)

        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError("Requested output from out-of-bound layer "
                             "output_layer={} (n_layers={})"
                             .format(output_layer, self.n_layers))

        z_hat = z0
        # Compute the following layers
        for layer_params in self.layers_parameters[:output_layer]:
            if 'threshold' in layer_params:
                th = layer_params['threshold']
            else:
                th = layer_params.get('step_size', 1/self.L)
            step_size = layer_params.get('step_size', 1.)
            if self.parametrization == "lista":
                if z_hat is None:
                    z_hat = x.matmul(layer_params['Wx'])
                else:
                    z_hat = z_hat.matmul(layer_params['Wz']) \
                        + x.matmul(layer_params['Wx'])
            else:
                if "W_coupled" in layer_params:
                    W = layer_params['W_coupled']
                elif "W_hessian" in layer_params:
                    W = self.D_.t().matmul(layer_params['W_hessian'])
                elif self.parametrization == "alista":
                    W = self.W
                else:
                    W = self.D_.t()
                W = W * step_size
                if z_hat is None:
                    z_hat = x.matmul(W)
                else:
                    res = z_hat.matmul(self.D_) - x
                    z_hat = z_hat - res.matmul(W)

            z_hat = soft_thresholding(z_hat, lmbd, th)

        return z_hat

    def _init_network_parameters(self, initial_parameters=[]):
        """ Initialize the parameters of the network. """
        if self.parametrization == "alista":
            self.W = check_tensor(get_alista_weights(self.D).T,
                                  device=self.device)

        n_atoms = self.D.shape[0]
        I_k = np.eye(n_atoms)

        parameters_config = PARAMETRIZATIONS[self.parametrization]

        self.layers_parameters = []
        for layer in range(self.n_layers):
            if len(initial_parameters) > layer:
                layer_params = initial_parameters[layer]
            else:
                if self.parametrization == "step":
                    layer_params = dict(step_size=np.array(1 / self.L))
                else:
                    layer_params = {}
                    if self.learn_th and 'threshold' in parameters_config:
                        layer_params['threshold'] = np.ones(n_atoms) / self.L
                    if self.parametrization == "lista":
                        layer_params['Wz'] = I_k - self.B / self.L
                        layer_params['Wx'] = self.D.T / self.L
                    elif self.parametrization == "coupled":
                        layer_params['W_coupled'] = self.D.T / self.L
                    elif self.parametrization == "first_step":
                        if layer == 0:
                            layer_params['W_coupled'] = self.D.T / self.L
                            layer_params['threshold'] = np.array(1 / self.L)
                        else:
                            layer_params['step_size'] = np.array(1 / self.L)
                            del layer_params['threshold']
                    elif self.parametrization == "alista":
                        layer_params['step_size'] = np.array(1 / self.L)
                        layer_params['threshold'] = np.array(1 / self.L)
                    elif self.parametrization == "hessian":
                        layer_params['W_hessian'] = I_k / self.L
                    else:
                        raise NotImplementedError()

            # transform all the parameters to learnable Tensor
            layer_params = {
                k: torch.nn.Parameter(check_tensor(p, device=self.device))
                for k, p in layer_params.items()}

            # Retrieve parameters hooks and register them

            for name, p in layer_params.items():
                self.register_parameter("layer{}-{}".format(layer, name), p)
                for h in parameters_config[name]:
                    self.pre_gradient_hooks[h].append(p)

            self.layers_parameters += [layer_params]
