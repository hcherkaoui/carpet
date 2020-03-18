""" Usefull functions for optimization."""
import time
import numpy as np
from scipy.optimize.linesearch import line_search_armijo


def condatvu(grad, obj, prox, psi, adj_psi, y0, x0, lbda, sigma, tau, rho=1.0,
             max_iter=1000, early_stopping=True, eps=np.finfo(np.float64).eps,
             times=False, debug=False, name='Optimisation', verbose=0):
    """ Condat-Vu algorithm. """
    # init the iterates
    y_old, x_old = np.copy(y0), np.copy(x0)

    # saving variables
    dg, pobj_, times_ = [0.0], [obj(x_old)], [0.0]

    for ii in range(max_iter):

        if times:
            t0 = time.process_time()

        # primal descent
        x_prox = x_old - tau * grad(x_old) - tau * adj_psi(y_old)

        # dual ascent
        y_tmp = y_old + sigma * psi(2 * x_prox - x_old)
        y_prox = y_tmp - sigma * prox(y_tmp / sigma)

        # relaxed updates
        x_new = rho * x_prox + (1 - rho) * x_old
        y_new = rho * y_prox + (1 - rho) * y_old

        # var. update
        x_old = x_new
        y_old = y_new

        # savings
        if times:
            times_.append(time.process_time() - t0)

        if debug:
            pobj_.append(obj(x_new))
            dg.append(np.linalg.norm(psi(x_new) - y_new))

        # printing
        if debug and verbose > 0:
            print(f"[{name}] Iteration {ii + 1} / {max_iter}, "
                  f"loss = {pobj_[ii]:.6e}, dg = {dg[ii]:.3e},")

        # early-stopping
        if early_stopping and dg[-1] < eps:
            break

    if not times and not debug:
        return x_new, y_new
    if times and not debug:
        return x_new, y_new, times_
    if not times and debug:
        return x_new, y_new, pobj_
    if times and debug:
        return x_new, y_new, pobj_, times_


def fista(grad, obj, prox, x0, momentum='fista', restarting=None, max_iter=100,
          step_size=None, early_stopping=True, eps=np.finfo(np.float64).eps,
          times=False, debug=False, verbose=0, name="Optimization"):
    """ ISTA like algorithm.
    """
    # parameters checking
    if verbose and not debug:
        print(f"[{name}] Can't have verbose if cost-func is not computed, "
              f"enable it by setting debug=True")

    if momentum not in ['fista', 'ista', 'greedy']:
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
    pobj_, times_, diff_ = [obj(y)], [0.0], [0.0]
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
        if momentum == 'fista':
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_old**2))
            y = x + (t_old - 1.0) / t * (x - x_old)

        elif momentum == 'greedy':
            y = x + (x - x_old)

        elif momentum == 'ista':
            y = x

        diff_.append(np.linalg.norm(x - x_old))

        # savings times
        if times:
            # skip cost-function computation for benchmark
            delta_t = time.time() - t0

        # savings cost-function values
        if debug:
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
        return y, np.array(pobj_)
    if times and debug:
        return y, np.array(pobj_), np.array(times_)
