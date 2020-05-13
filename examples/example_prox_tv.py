import torch
import numpy as np
import matplotlib.pyplot as plt
from carpet.checks import check_tensor
from carpet.proximity_tv import ProxTV, RegTV


def loss_subgradient(z, c, lmbd):
    # Gradient of the loss is a sub-gradient
    res = (z - c)
    loss = (res * res).sum(axis=1) / 2
    loss += lmbd * torch.abs(z[:, 1:] - z[:, :-1]).sum(axis=1)
    return loss


def loss_prox(z, c, lmbd):
    # Use prox to compute gradient of the loss

    res = (z - c)
    loss = (res * res).sum(axis=1) / 2
    loss = RegTV.apply(loss, z, lmbd)
    return loss


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description="""Regress a function based on its prox. Try to solve:
        min_x ||c - prox(z)||_2^2 + lambda||prox_z||_TV

        Solutions for this problem are z* such that prox(z*) = prox(c).
        """)
    parser.add_argument('--starting-point', '-s', type=str, default='random',
                        help="Initialisation for x. should be one of"
                        "{'random', 'perturb', 'mean'}")
    parser.add_argument('--lmbd', type=float, default=.5,
                        help="Regularization parameter.")
    args = parser.parse_args()

    n_samples = 1
    n_atoms = 10
    n_dim = 5
    lmbd = args.lmbd
    max_iter = 100

    params = {
        'prox': dict(loss=loss_prox, color='b'),
        'subgradient': dict(loss=loss_subgradient, color='g'),
    }

    # Layer with backprop for prox_tv
    prox = ProxTV(prox='prox_tv', n_dim=n_dim)

    # get a random vector that we try to regress
    x = torch.randn(n_samples, n_atoms)
    lmbd_ = check_tensor(lmbd)

    # compute the starting point
    if args.starting_point == 'mean':
        z0 = x.numpy().mean() * np.ones(x.shape)
    elif args.starting_point == 'random':
        z0 = np.random.randn(1, n_atoms)
    elif args.starting_point == 'perturb':
        z0 = x.numpy() + .1 * np.random.randn(*x.shape)
    else:
        raise NotImplementedError()

    # Compute and display the prox of c (our target)
    prox_x = prox(x, lmbd_)
    ax = plt.subplot(111)
    ax.plot(x.data[0], 'r', linewidth=2)
    ax.plot(prox_x[0].data, 'r--', linewidth=2)
    _, ax_loss = plt.subplots()

    max_iter = 100
    for k, param in params.items():
        # Initiate optim vars
        lr = 1000
        loss_func = param['loss']
        c = param['color']
        z_ = check_tensor(z0, requires_grad=True)
        log = [loss_func(prox(z_, lmbd_), x, lmbd_).detach().numpy()]
        for i in range(max_iter):

            # Compute loss and gradient
            z_.grad = None
            output = prox(z_, lmbd_)
            loss = loss_func(output, x, lmbd_)
            loss.backward()
            log.append(loss.detach().numpy())
            grad_z = z_.grad

            # Backtracking line-search to make sure we get a deacreasing
            # loss function
            z_.data -= lr * grad_z
            with torch.no_grad():
                loss = loss_func(prox(z_, lmbd_), x, lmbd_)
                while loss.detach().numpy() >= log[-1] and lr > 1e-20:
                    lr /= 2
                    z_.data += lr * grad_z
                    loss = loss_func(prox(z_, lmbd_), x, lmbd_)
                # Break if we reached the limit step size
                if lr <= 1e-20:
                    print(f"converged in {i} iterations. "
                          f"||g|| = {(grad_z*grad_z).sum()}\n"
                          f"nabla_z={grad_z}")
                    with torch.enable_grad():
                        g, = torch.autograd.grad(
                            loss_func(output, x, lmbd_), output)
                        print(f"||dL/d(prox z)|| = {(g*g).sum()}")

                        output = prox(z_, lmbd_)
                        loss = loss_func(output, x, lmbd_)
                        loss.backward()

                    ax.plot(z_[0].data, f'{c}-', alpha=1)
                    ax.plot(output[0].data, f'{c}--', alpha=1)
                    break
                lr *= 10000

            if i % 10 == 0:
                ax.plot(z_[0].data, f'{c}-', alpha=i/max_iter)
                ax.plot(output[0].data, f'{c}--', alpha=i/max_iter)

        cost = np.array(log)
        c_star = loss_func(prox_x.data, x.data, lmbd).data.numpy()
        cost -= c_star + 1e-10
        ax_loss.semilogy(cost, color=c, label=k)
    plt.legend()
    plt.show()
