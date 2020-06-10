import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


plt.close('all')

# files = {
#     .1: 'loss_comparison_2_2020-05-28_16h38_0.1_427.pkl',
#     .8: 'loss_comparison_2_2020-05-28_22h08_0.8_427.pkl',
# }
files = {
    .1: 'loss_comparison_3_2020-06-02_22h08_0.1_427.pkl',
    .8: 'loss_comparison_3_2020-06-02_22h08_0.8_427.pkl',
}


EPS = 1e-10
LINEWIDTH = 4


STYLE_METHODS = {
    'fista_synthesis': {
        'color': 'C0', 'ls': '--',
        'marker': '*', 'label': 'FISTA - synthesis'
    },
    'lista_synthesis': {
        'color': 'C0', 'ls': '-',
        'marker': 's', 'label': 'LISTA - synthesis'
    },
    'ista_analysis': {
        'color': 'C1', 'ls': '--',
        'marker': 'o', 'label': 'PGD - analysis'
    },
    'fista_analysis': {
        'color': 'C1', 'ls': '--',
        'marker': '*', 'label': 'Accelerated PGD - analysis'
    },
    'lpgd_taut': {
        'color': 'C3', 'ls': '-',
        'marker': 's', 'label': 'LPGD-Taut'
    },
    'lpgd_lista_per-layer_50': {
        'color': 'C3', 'ls': '-',
        'marker': '^', 'label': 'LPGD-LISTA[50]'
    },
}

# Setup matplotlib fonts
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amssymb}']
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 18

n_plots = 2
id_row = 0
fig_compare = plt.figure(figsize=(6.4 * n_plots, 3.6))
fig_prox, axes = plt.subplots(ncols=2, figsize=(6.4 * n_plots, 2.8))
gs = mpl.gridspec.GridSpec(nrows=2, ncols=n_plots,
                           height_ratios=[.1, .9])
axis_test = None
ax = None


for k, (lmbd, filename) in enumerate(files.items()):
    df = pd.read_pickle(f'outputs_plots/{filename}')

    ref = df.query("key == 'reference'")
    df = df.query("key != 'reference'")
    C0 = ref.loc[ref['n_layers'].idxmin()].test_loss
    ref = ref.loc[ref['n_layers'].idxmax()]
    seed = df.iloc[0]['seed']
    lmbd = df.iloc[0]['lmbd']

    axis_test = fig_compare.add_subplot(gs[1, k])  # , sharey=axis_test)
    handles_compare = {}

    c_star_test = ref.test_loss - EPS
    for method, style in STYLE_METHODS.items():
        this_loss = df.query("key == @method")
        if ':' in style['color']:
            style['color'] = style['color'].split(':')[1]
        ticks_layers = np.r_[0, this_loss['n_layers']]
        test_loss = np.r_[C0, this_loss.test_loss] - c_star_test
        handles_compare[style['label']] = axis_test.loglog(
            ticks_layers + 1, test_loss, **style, lw=LINEWIDTH, ms=3*LINEWIDTH
        )[0]

    # Formatting test loss
    axis_test.grid()
    axis_test.set_xlabel("Layers $t$")
    if k == 0:
        axis_test.set_ylabel(
            r"$\mathbb E\left[P_{x}(u^{(t)}) - P_{x}(u^*)\right]$"
        )

    axis_test.set_xticks(ticks_layers + 1)
    axis_test.set_xticklabels(ticks_layers)

    fig_compare.tight_layout()

    ax = axes[k]
    handles = {'Trained': '', 'Untrained': ''}
    for i, nl in enumerate([20, 50]):
        handles[f'{nl} inner layers'] = plt.Line2D([], [], color=f'C{i}',
                                                   lw=LINEWIDTH)
        for ls, learn, label in [('--', 'none', 'Untrained'),
                                 ('-', 'per-layer', 'Trained')]:
            curve = []
            method = df.query(f"key == 'lpgd_lista_{learn}_{nl}'")
            for _, v in method.iterrows():
                prox_tv_loss = v[f'prox_tv_loss_test']
                if len(prox_tv_loss) == 0:
                    continue
                curve.append((v['n_layers'], prox_tv_loss[-1]))
                # ax.plot(prox_tv_loss, color=f'C{i}', ls=ls, alpha=.1,
                #         lw=LINEWIDTH)
            curve = np.array(curve).T
            ax.loglog(curve[0], curve[1], color=f'C{i}', ls=ls,
                      lw=LINEWIDTH)
            handles[label] = plt.Line2D([], [], color='k', ls=ls, lw=LINEWIDTH)
    ax.set_xlabel('Layers $t$')
    if k == 0:
        ax.set_ylabel(
            r'$\mathbb E\left[F_{u^{(t)}}(z^{(L)}) - F_{u^{(t)}}(z^*)\right]$'
        )
    ax.grid()
    ax.set_xlim(1, 40)
    ax.set_xticks(ticks_layers + 1)
    ax.set_xticklabels(ticks_layers)

ax_legend = fig_compare.add_subplot(gs[0, :])
ax_legend.set_axis_off()
ax_legend.legend(
    handles_compare.values(), handles_compare.keys(), ncol=3, loc='center',
    bbox_to_anchor=(0, .95, 1, .05), fontsize=18
)
fig_compare.subplots_adjust(left=.08, bottom=.1, right=.99, top=.9,
                            wspace=.15, hspace=.15)
fig_compare.savefig('outputs_plots/loss_comparison.pdf', dpi=300,
                    bbox_inches='tight', pad_inches=0)

# ax_legend = fig_prox.add_subplot(gs[0, :])
# ax_legend.set_axis_off()
axes[0].legend(
    handles.values(), handles.keys(), ncol=2,
    # loc='center', bbox_to_anchor=(0, .95, 1, .05),
    fontsize=18
)
# plt.legend(handles.values(), handles.keys(), ncol=2,
#            loc='lower left', bbox_to_anchor=(-0.08, 1, 1, .05))
fig_prox.subplots_adjust(left=.08, bottom=.1, right=.99, top=.93,
                         wspace=.15, hspace=.1)
fig_prox.savefig('outputs_plots/comparison_prox_tv_loss.pdf', dpi=300,
                 bbox_inches='tight', pad_inches=0)


plt.show()
