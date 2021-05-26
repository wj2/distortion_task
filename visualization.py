
import numpy as np
import matplotlib.pyplot as plt

import general.plotting as gpl
import general.utility as u
import distortion_task.analysis as da

def plot_corrects(corrs, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for i, c in enumerate(corrs):
        blocks = np.arange(1, c.shape[1] + 1)
        gpl.plot_trace_werr(blocks, c, ax=ax, conf95=True)

def plot_model_comparison_blocks(results, axs=None, norm=True, n_boots=1000,
                                 rm_nan=True, func=None, chance_line=0,
                                 rm_warning=True, **kwargs):
    if func is None:
        func = lambda x: np.nanmean(x)
    if axs is None:
        f, axs = plt.subplots(1, 2)
    halfs = np.zeros((len(results), n_boots))
    for i, br in enumerate(results):
        plot_model_comparison(br, ax=axs[0], norm=norm, **kwargs)
        vals = da.get_model_comparison_distrib(br, norm=norm,
                                               rm_warning=rm_warning)
        if rm_nan:
            vals = vals[np.logical_not(np.isnan(vals))]
        halfs[i] = u.bootstrap_list(vals, func, n=n_boots)
    gpl.plot_trace_werr(np.arange(halfs.shape[0]), halfs.T, ax=axs[1],
                        conf95=True)
    gpl.add_hlines(chance_line, axs[1])
        
def plot_model_comparison(block_results, ax=None, norm=False, **kwargs):
    vals = da.get_model_comparison_distrib(block_results, norm=norm)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.hist(vals, **kwargs)
    return ax
        
def plot_generalization(xs, decs, ax=None, v_line=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for dec in decs:
        gpl.plot_trace_werr(xs, dec, ax=ax)
    gpl.plot_trace_werr(xs, decs, ax=ax, conf95=True)
    if v_line is not None:
        gpl.add_vlines(v_line, ax)

def plot_multiple_scatter(decs_b, bhv_b, axs=None):
    f, axs = plt.subplots(1, 2)
    corr_traj = np.zeros((len(decs_b), 3))
    for i, dec in enumerate(decs_b):
        bhv = bhv_b[i]
        axs[0].plot(dec, bhv, 'o')
        dec = np.mean(dec, axis=1)
        corr_traj[i] = gpl.get_corr_conf95(dec, bhv)
    bs = np.arange(len(decs_b))
    l = axs[1].plot(bs, corr_traj[:, 0])
    axs[1].fill_between(bs, corr_traj[:, 1], corr_traj[:, 2],
                        color=l[0].get_color(), alpha=.1)
    return axs
        
def plot_masked_scatter(decs, bhv, t_ind, decs_tr=None, func=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)    
    m_dec = da.compute_decoding_mask(decs[..., t_ind])
    m_bhv = da.compute_performance_mask(bhv)
    mask = np.logical_and(m_dec, m_bhv)
    if func is not None:
        decs = func(decs, decs_tr)
    decs_m = decs[mask]
    bhv_m = bhv[mask]
    cent, lower, upper = gpl.get_corr_conf95(decs_m, bhv_m)
    ax.plot(decs_m, bhv_m, 'o')
    print(cent, lower, upper)
    return ax

def plot_confusability(c1, c2, sep=1, scale=.1, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    c1_offs = np.random.randn(c1.shape[0])*scale
    c2_offs = np.random.randn(c2.shape[0])*scale + sep
    for i, c1_i in enumerate(c1):
        c2_i = c2[i]
        gpl.plot_conf_interval(c1_offs[i], c1_i, ax=ax)
        gpl.plot_conf_interval(c2_offs[i], c2_i, ax=ax)
    gpl.add_hlines(0, ax)
    
def plot_scatter(q1, q2, t_ind, line_x=None, line_y=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.plot(q1[:, t_ind], q2[:, t_ind], 'o')
    if line_x is not None and line_y is None:
        gpl.add_vlines(line_x, ax)
    elif line_x is None and line_y is not None:
        gpl.add_hlines(line_y, ax)
    elif line_x is not None and line_y is not None:
        ax.plot(line_x, line_y)
    return ax
