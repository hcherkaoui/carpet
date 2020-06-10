import numpy as np

from prox_tv import tv1_1d
from carpet.optimization import fista
from carpet.utils import init_vuz, v_to_u
from carpet.proximity import pseudo_soft_th_numpy
from carpet.loss_gradient import (analysis_primal_obj, analysis_primal_grad,
                                  synthesis_primal_grad, synthesis_primal_obj)


class _IterativeSolver:

    def __init__(self, device=None, verbose=1, name='Iterativesolver'):

        if device is not None and 'cuda' in device:
            import warnings
            warnings.warn("Cannot use LpgdTautString on cuda device. "
                          "Falling back to CPU.")

        self.name = name
        self.verbose = verbose

    def fit(self, x_train, lbda):
        pass

    def export_parameters(self):
        return None

    def transform_to_u(self, x, lbda, output_layer=None):
        """Compute the output in primal analysis from given x and lbda

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            input of the network.
        lbda: float
            Regularization level for the optimization problem.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. If set to None, output the last layer of the
            network
        """
        output = self.transform(x, lbda, output_layer=None)
        if self._output == 'u-analysis':
            return output
        if self._output == 'z-synthesis':
            return np.cumsum(output, axis=-1)

        assert self._output == 'v-analysis_dual'
        return v_to_u(output, x, lbda, A=self.A, D=self.D)


class IstaSynthesis(_IterativeSolver):
    _output = 'z-synthesis'

    def __init__(self, A, n_layers, momentum='fista', initial_parameters=None,
                 device=None, verbose=1):

        self.n_layers = n_layers
        self.momentum = momentum

        n_atoms = A.shape[0]
        self.A = A
        self.L = np.triu(np.ones((n_atoms, n_atoms)))
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        LA = self.L.dot(A)
        self.step_size = 1.0 / np.linalg.norm(LA, ord=2) ** 2

        super().__init__(device=device, verbose=verbose, name="IstaSynthesis")

    def transform(self, x, lbda, output_layer=None):
        if output_layer is None:
            output_layer = self.n_layers

        _, _, z0 = init_vuz(self.A, self.D, x)

        params = dict(
            grad=lambda z: synthesis_primal_grad(z, self.A, self.L, x),
            obj=lambda z: synthesis_primal_obj(z, self.A, self.L, x, lbda),
            prox=lambda z, mu: pseudo_soft_th_numpy(z, lbda, mu),
            x0=z0,  momentum=self.momentum, step_size=self.step_size,
            restarting=None, max_iter=output_layer, early_stopping=False,
            debug=True, verbose=self.verbose
        )

        return fista(**params)[0]


class IstaAnalysis(_IterativeSolver):
    _output = 'u-analysis'

    def __init__(self, A, n_layers, momentum='fista', initial_parameters=None,
                 device=None, verbose=1):
        self.n_layers = n_layers
        self.momentum = momentum

        n_atoms = A.shape[0]
        self.A = A
        self.L = np.triu(np.ones((n_atoms, n_atoms)))
        self.D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]

        self.step_size = 1.0 / np.linalg.norm(A, ord=2) ** 2

        super().__init__(device=device, verbose=verbose, name="IstaAnalysis")

    def fit(self, x_train, lbda):
        pass

    def transform(self, x, lbda, output_layer=None):
        if output_layer is None:
            output_layer = self.n_layers

        _, u0, _ = init_vuz(self.A, self.D, x)

        params = dict(
            grad=lambda z: analysis_primal_grad(z, self.A, x),
            obj=lambda z: analysis_primal_obj(z, self.A, self.D, x, lbda),
            prox=lambda z, s: np.array([tv1_1d(z_, lbda * s) for z_ in z]),
            x0=u0,  momentum=self.momentum, step_size=self.step_size,
            restarting=None, max_iter=output_layer, early_stopping=False,
            debug=True, verbose=self.verbose,
        )

        return fista(**params)[0]
