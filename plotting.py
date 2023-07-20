import numpy as np
import matplotlib.pyplot as plt

def load_figure_data(dataset:str):
    """Load figure data from paper."""

    f1_rtkm = np.load('figure_data/f1_rtkm_'+ dataset + '.npy')
    f1_kmor = np.load('figure_data/f1_kmor_'+ dataset + '.npy')
    f1_neo = np.load('figure_data/f1_neo_'+ dataset + '.npy')

    me_rtkm = np.load('figure_data/me_rtkm_'+ dataset + '.npy')
    me_kmor = np.load('figure_data/me_kmor_'+ dataset + '.npy')
    me_neo = np.load('figure_data/me_neo_'+ dataset + '.npy')

    min_rtkm = np.load('figure_data/min_rtkm_'+ dataset + '.npy')
    min_kmor = np.load('figure_data/min_kmor_'+ dataset + '.npy')
    min_neo = np.load('figure_data/min_neo_'+ dataset + '.npy')

    max_rtkm = np.load('figure_data/max_rtkm_'+ dataset + '.npy')
    max_kmor = np.load('figure_data/max_kmor_' + dataset + '.npy')
    max_neo = np.load('figure_data/max_neo_' + dataset + '.npy')

    alpha_vals = np.load('figure_data/alpha_vals_' + dataset + '.npy')

    return f1_rtkm, f1_kmor, f1_neo, me_rtkm, me_kmor, me_neo, \
           min_rtkm, min_kmor, min_neo, max_rtkm, max_kmor, max_neo, alpha_vals


def make_figure(dataset:str, f1_rtkm=None, f1_kmor=None, f1_neo=None, me_rtkm=None, me_kmor=None, me_neo=None,
                min_rtkm=None, min_kmor=None, min_neo=None, max_rtkm=None, max_kmor=None, max_neo=None,
                alpha_vals=None) -> None:

    if f1_rtkm or f1_kmor or f1_neo or me_rtkm or me_kmor or me_neo or \
            min_rtkm or min_kmor or min_neo or max_rtkm or max_kmor or max_neo or alpha_vals is None:

        f1_rtkm, f1_kmor, f1_neo, me_rtkm, me_kmor, me_neo, \
        min_rtkm, min_kmor, min_neo, max_rtkm, max_kmor, max_neo, alpha_vals = load_figure_data(dataset)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    ax1.plot(alpha_vals, f1_rtkm, label='RTKM', c='r')
    ax1.plot(alpha_vals, f1_kmor, label='KMOR', c='b')
    ax1.plot(alpha_vals, f1_neo, label='NEO', c='g')
    ax1.fill_between(alpha_vals, min_rtkm[0, :], max_rtkm[0, :], color='r', alpha=0.2)
    ax1.fill_between(alpha_vals, min_kmor[0, :], max_kmor[0, :], color='b', alpha=0.2)
    ax1.fill_between(alpha_vals, min_neo[0, :], max_neo[0, :], color='g', alpha=0.2)
    ax1.set_title('Sensitivity of $F_1$ score')
    ax1.set_ylabel('Average $F_1$ score')
    ax1.yaxis.label.set_fontsize(25)
    ax1.title.set_fontsize(30)
    ax1.tick_params(axis='y', labelsize=15)

    ax2.plot(alpha_vals, me_rtkm, label='RTKM', c='r')
    ax2.plot(alpha_vals, me_kmor, label='KMOR', c='b')
    ax2.plot(alpha_vals, me_neo, label='NEO', c='g')
    ax2.fill_between(alpha_vals, min_rtkm[1, :], max_rtkm[1, :], color='r', alpha=0.2)
    ax2.fill_between(alpha_vals, min_kmor[1, :], max_kmor[1, :], color='b', alpha=0.2)
    ax2.fill_between(alpha_vals, min_neo[1, :], max_neo[1, :], color='g', alpha=0.2)
    ax2.set_title('Sensitivity of $M_e$ score')
    ax2.set_ylabel('Average $M_e$ score')
    ax2.set_xlabel('Predicted Percentage of Outliers ' r'$\alpha$')
    ax2.xaxis.label.set_fontsize(25)
    ax2.yaxis.label.set_fontsize(25)
    ax2.title.set_fontsize(30)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.legend(fontsize=20)
    plt.savefig(dataset+'.pdf', format='pdf')

    return None