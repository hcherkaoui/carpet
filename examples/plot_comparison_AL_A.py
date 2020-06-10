import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}'
]

lw = 4
n_samples = 1000
N = np.logspace(1, 3, 20, dtype=int)

OUTPUT_DIR = pathlib.Path('outputs_plots')


filename = OUTPUT_DIR / 'comparison_AL_A.pkl'
if filename.exists():
    df = pd.read_pickle(filename)
else:
    curve = []
    for n in N:
        print(f"\rSize {n} :", end='', flush=True)
        a, e = [], []
        L = np.tri(n)
        for i in range(n_samples):
            print(f"{i / n_samples:7.2%}" + "\b" * 7, end='', flush=True)
            A = np.random.randn(n, n)
            e.append(np.linalg.norm(A, ord=2) ** 2)
            a.append(np.linalg.norm(A.dot(L), ord=2) ** 2)
        r = np.array(a)/e
        curve.append({
            'n': n,
            'mean': r.mean(),
            'median': np.median(r),
            'q1': np.quantile(r, q=.1),
            'q9': np.quantile(r, q=.9),
            'b': 1 / (32 * np.cos(n*np.pi/(2*n+1)) ** 2),
            'lb': (n+1/2) / np.pi ** 2
        })
    df = pd.DataFrame(curve)
    df.to_pickle(filename)
print('done'.ljust(40))


plt.figure(figsize=(6.4, 2.4))
plt.loglog(df['n'], df['mean'],
           label=r'Mean $\mathbb E\left[\frac{\|AL\|_2^2}{\|A\|_2^2}\right]$',
           lw=lw)

plt.fill_between(df['n'], df['q1'], df['q9'],
                 alpha=.3, color='C0')

plt.loglog(df.n, df.lb, 'C2', label='Proposition 2.1', lw=lw)
plt.loglog(df.n, df.b, 'C1--', label='Conjecture 2.2', lw=lw - 1)
plt.legend(loc='lower left', bbox_to_anchor=(-0.17, 1, .05, 1), ncol=3,
           fontsize=16, columnspacing=1, handlelength=1.5)
plt.xlim(N.min(), N.max())
plt.grid()
plt.xlabel("Dimension $k$")
plt.ylabel(r"$\|AL\|^2_2 / \|A\|^2_2$")
plt.savefig('outputs_plots/comparison_AL_A.pdf', dpi=300)
plt.show()
