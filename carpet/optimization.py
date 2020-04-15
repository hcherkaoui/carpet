""" Usefull functions for optimization."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import numpy as np
from scipy.optimize.linesearch import line_search_armijo


def condatvu(grad, obj, prox, psi, adj_psi, v0, z0, lbda, sigma, tau, rho=1.0,
             max_iter=1000, early_stopping=True, eps=np.finfo(np.float64).eps,
             times=False, debug=False, name='Optimisation', verbose=0):
    """ Condat-Vu algorithm. """
    # init the iterates
    v_old, z_old = np.copy(v0), np.copy(z0)

    # saving variables
    dg, pobj_, times_ = [np.linalg.norm(psi(z0) - v0)], [obj(z_old)], [0.0]
    saved_z_,saved_v_ = [z0], [v0]

    for ii in range(max_iter):

        if times:
            t0 = time.process_time()

        # primal descent
        z_prox = z_old - tau * grad(z_old) - tau * adj_psi(v_old)

        # dual ascent
        v_tmp = v_old + sigma * psi(2 * z_prox - z_old)
        v_prox = v_tmp - sigma * prox(v_tmp / sigma)

        # relaxed updates
        z_new = rho * z_prox + (1 - rho) * z_old
        v_new = rho * v_prox + (1 - rho) * v_old

        # var. update
        z_old = z_new
        v_old = v_new

        # savings
        if times:
            times_.append(time.process_time() - t0)

        if debug:
            saved_z_.append(z_new)
            saved_v_.append(v_new)
            pobj_.append(obj(z_new))
            dg.append(np.linalg.norm(psi(z_new) - v_new))

        # printing
        if debug and verbose > 0:
            print(f"[{name}] Iteration {ii + 1} / {max_iter}, "  # noqa: E999
                  f"loss = {pobj_[ii]:.6e}, dg = {dg[ii]:.3e},")

        # early-stopping
        if early_stopping and dg[-1] < eps:
            break

    if not times and not debug:
        return z_new, v_new
    if times and not debug:
        return z_new, v_new, np.array(times_)
    if not times and debug:
        return z_new, v_new, saved_z_, saved_v_, np.array(pobj_)
    if times and debug:
        return z_new, v_new, saved_z_, saved_v_, np.array(pobj_), \
               np.array(times_)


def fista(grad, obj, prox, x0, momentum='fista', restarting=None, max_iter=100,
          step_size=None, early_stopping=True, eps=np.finfo(np.float64).eps,
          times=False, debug=False, verbose=0, name="Optimization"):
    """ ISTA like algorithm. """
    # parameters checking
    if verbose and not debug:
        print(f"[{name}] Can't have verbose if cost-func is not computed, "
              f"enable it by setting debug=True")

    if momentum not in [None, 'fista', 'greedy']:
        raise ValueError(f"[{name}] momentum should be ['fista', 'ista', "
                         f"'greedy'], got {momentum}")

    if restarting not in [None, 'obj', 'descent']:
        raise ValueError(f"[{name}] restarting should be [None, 'obj', "
                         f"'descent'], got {restarting}")

    if momentum == 'ista' and restarting in ['obj', 'descent']:
        raise ValueError(f"[{name}] restarting can't be set to 'obj' or "
                         f"'descent' if momentum == 'ista'")

    # prepare the iterate
    x_old, x, y, y_old = np.copy(x0), np.copy(x0), np.copy(x0), np.copy(x0)
    pobj_, times_, diff_, saved_y_ = [obj(y)], [0.0], [0.0], [y]
    t = t_old = 1

    # prepare the adaptative-step variables
    adaptive_step_size = False
    if step_size is None:
        adaptive_step_size = True
        step_size = 1.0
        old_fval = pobj_[0]

    # main loop
    for ii in range(max_iter):

        if times:
            t0 = time.time()

        grad_ = grad(y)

        # adaptative step-size
        if adaptive_step_size:
            step_size, _, old_fval = line_search_armijo(
                    obj, y.ravel(), -grad_.ravel(), grad_.ravel(),
                    old_fval, c1=1.0e-5, alpha0=step_size)
            if step_size is None:
                step_size = 0.0

        # main descent step
        x = prox(y - step_size * grad_, step_size)

        # fista acceleration
        if momentum is None:
            y = x

        elif momentum == 'fista':
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_old**2))
            y = x + (t_old - 1.0) / t * (x - x_old)

        elif momentum == 'greedy':
            y = x + (x - x_old)

        diff_.append(np.linalg.norm(x - x_old))

        # savings times
        if times:
            # skip cost-function computation for benchmark
            delta_t = time.time() - t0

        # savings cost-function values
        if debug:
            saved_y_.append(y)
            if adaptive_step_size:
                pobj_.append(old_fval)
            else:
                pobj_.append(obj(y))

        # savings times, restart after cost-function computation
        if times:
            t0 = time.time()

        if restarting == 'obj' and (pobj_[-1] > pobj_[-2]):
            # restart if cost function increase
            if momentum == 'fista':
                x = x_old
                t = 1.0
            elif momentum == 'greedy':
                y = x

        if restarting == 'descent' and np.sum((y_old - x) * (x - x_old)) > 0.0:
            # restart if x_k+1 - x_k has the same direction than x_k - x_k-1
            if momentum == 'fista':
                x = x_old
                t = 1.0
            elif momentum == 'greedy':
                y = x

        # variables updates k+1, k, k-1
        t_old = t
        x_old = x
        y_old = y

        # verbose at every 50th iterations
        if debug and verbose > 0 and ii % 100:
            print(f"[{name}] Iteration {100.0 * (ii + 1) / max_iter:.0f}%, "
                  f"loss = {pobj_[ii]:.3e}, "
                  f"grad-norm = {np.linalg.norm(grad_):.3e}")

        # early-stopping on || x_k - x_k-1 || < eps
        if diff_[-1] <= eps and early_stopping:
            if debug:
                print(f"\n[{name}] early-stopping "
                      f"done at {100.0 * (ii + 1) / max_iter:.0f}%, "
                      f"loss = {pobj_[ii]:.3e}, "
                      f"grad-norm = {np.linalg.norm(grad_):.3e}")
            break

        # divergence safeguarding
        if diff_[-1] > np.finfo(np.float64).max:
            raise RuntimeError(f"[{name}] algo. have diverged during.")

        # savings times
        if times:
            times_.append(delta_t + time.time() - t0)

    if not times and not debug:
        return y
    if times and not debug:
        return y, np.array(times_)
    if not times and debug:
        return y, saved_y_, np.array(pobj_)
    if times and debug:
        return y, saved_y_, np.array(pobj_), np.array(times_)
