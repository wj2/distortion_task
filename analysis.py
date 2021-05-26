
import numpy as np

import general.utility as u
import general.neural_analysis as na
import general.data_io as dio
import general.plotting as gpl

def get_correct_rates(data, block_field='block', correct_field='correct',
                      n_boots=1000):
    corrects = data[correct_field]
    blocks = data[block_field]
    out = []
    for i, bd in enumerate(blocks):
        blocks_i = np.unique(bd)
        bs = np.zeros((n_boots, len(blocks_i)))
        for j, bi in enumerate(blocks_i):
            corrs = np.array((corrects[i][bd == bi]))
            bs[:, j] = u.bootstrap_list(corrs, np.mean, n=n_boots)
        out.append(bs)
    return out


def get_first_stim_occurences(data, stimfield='sample'):
    stims = data[stimfield]
    mask = []
    for stim_seq in stims:
        us = np.unique(stim_seq)
        mask_ss = np.zeros(len(stim_seq), dtype=bool)
        for j, u in enumerate(us):
            mask_ss[np.argmax(stim_seq == u)] = True
        mask.append(mask_ss)
    rs = dio.ResultSequence(mask)
    return rs

def compute_session_confusability(session_info, t_ind, n_boots=1000,
                                  min_trls=1):
    out = np.zeros((2, len(session_info), n_boots))
    for i, si in enumerate(session_info):
        if si[0].shape[0] > min_trls:
            out[:, i] = compute_confusability(*si, t_ind, n_boots=n_boots)
        else:
            out[:, i] = np.nan
    return out        



def compute_confusability(corr, cat, dec, t_ind, n_boots=1000):
    cat1_err = np.logical_and(corr == 0, cat == 0)
    cat2_err = np.logical_and(corr == 0, cat == 1)
    cat1_corr = np.logical_and(corr == 1, cat == 0)
    cat2_corr = np.logical_and(corr == 1, cat == 1)
    dec = dec[0, 0, :, t_ind]
    c1err_mean = u.bootstrap_list(dec[cat1_err], np.mean, n=n_boots)
    c2err_mean = u.bootstrap_list(dec[cat2_err], np.mean, n=n_boots)
    c1corr_mean = u.bootstrap_list(dec[cat1_corr], np.mean, n=n_boots)
    c2corr_mean = u.bootstrap_list(dec[cat2_corr], np.mean, n=n_boots)
    return c1corr_mean - c2corr_mean, c1err_mean - c2err_mean

def compute_generalization_rate(data, **kwargs):
    fs = get_first_stim_occurences(data)
    data_fs = data.mask(fs)
    out = get_correct_rates(data_fs, **kwargs)
    return out

def compute_performance_mask(corr_arr):
    corr_m = np.mean(corr_arr.T, axis=0)
    perf_conf = gpl.conf95_interval(corr_arr.T) + corr_m
    mask_bhv = perf_conf[1] > .5
    return mask_bhv

def compute_decoding_mask(decs, thr=.65):
    if len(decs.shape) > 1:
        out = compute_performance_mask(decs)
    else:
        out = decs > thr
    return out

def compute_bhv_dec_mask(dec_tr, corr_arr, thr=.65):
    m_bhv = compute_performance_mask(corr_arr)
    m_dec = compute_decoding_mask(dec_tr, thr=thr)
    mask = np.logical_and(m_bhv, m_dec)
    return mask

def explain_block_generalization_responses(oms, t_ind, **kwargs):
    outs = []
    for i, om_i in enumerate(oms):
        outs.append([])
        for j, om_ij in enumerate(om_i):
            corr, cat, dec_val = om_ij
            dec_val_use = dec_val[0, 0, :, t_ind]
            out = explain_generalization_response(corr, cat, dec_val_use,
                                                  **kwargs)
            outs[i].append(out)
    return outs

def get_model_comparison_distrib(block_results, norm=False,
                                 rm_warning=True):
    vals = []
    for i, session in enumerate(block_results):
        session_val = get_model_comparison_value(session[2], norm=norm,
                                                 rm_warning=rm_warning)
        vals.append(session_val)
    vals = np.array(vals)
    return vals

def get_model_comparison_value(comp, norm=False, rm_warning=True):
    if rm_warning and np.any(comp['warning']):
        val = np.nan
    else:
        val = comp['d_loo']['null'] - comp['d_loo']['logit']
        if norm:
            val = val/comp['dse'][1]
    return val

def explain_generalization_response(corr, cat, dec_val, **kwargs):
    resp = np.zeros_like(corr)
    resp[cat == 0] = corr[cat == 0] == 1
    resp[cat == 1] = corr[cat == 1] == 0
    out = na.fit_logit(dec_val, resp, **kwargs)
    return out

def combine_block_generalization(outs, t_ind, func=None, thr=.65,
                                 exclude_gen_trls=None, combine=False,
                                 mean_resamples=True):
    dec_perfs = []
    bhv_perfs = []
    oms = []
    for i, (otr, ogen, _, corr_arr, om) in enumerate(outs):
        otr_m = np.mean(otr, axis=1)
        mask = compute_bhv_dec_mask(otr_m[:, t_ind], corr_arr, thr=thr)
        if func is not None:
            ogen = func(ogen, otr)
        if exclude_gen_trls is not None:
            n_gens = np.array(list(om_i[-1].shape[2] for om_i in om))
            m_gen_trls = n_gens > exclude_gen_trls
            mask = np.logical_and(mask, m_gen_trls)
        dec_perfs.append(ogen[mask, :, t_ind])
        bhv_perfs.append(np.mean(corr_arr[mask], axis=1))
        oms.append(np.array(om, dtype=object)[mask])
    if combine:
        dec_perfs = np.concatenate(dec_perfs)
        bhv_perfs = np.concatenate(bhv_perfs)
        oms = np.concatenate(oms)
    return dec_perfs, bhv_perfs, oms

def compute_generalization_blocks(data, *args, block_field='block', **kwargs):
    max_blocks = np.max(list(map(np.max, data[block_field])))
    outs = []
    for i in range(1, max_blocks):
        out = generalize_blocks(data, i, *args, block_field=block_field,
                                other_info_fields=['date', 'animal'],
                                **kwargs)
        outs.append(out)
    return outs

def generalize_blocks(data, training_blocks, tbeg, tend, binsize, binstep,
                      block_field='block', outcome_field='outcome',
                      category_field='category', folds_n=10,
                      correct_field='correct', n_boots=1000,
                      min_trls=100, equal_trials=False,
                      resamples=10, next_block=False,
                      other_info_fields=None, **kwargs):
    n_blocks = np.array(list(map(np.max, data[block_field])))
    if training_blocks < 1:
        training_blocks = np.round(training_blocks*n_blocks).astype(int)
    smask = n_blocks >= training_blocks + 1
    data = data.session_mask(smask)
    if other_info_fields is not None:
        other_info = data[other_info_fields]
    train_mask = data[block_field] <= training_blocks
    if next_block:
        test_mask = data[block_field] == training_blocks + 1
    else:
        test_mask = data[block_field] > training_blocks
    fs = get_first_stim_occurences(data)
    test_mask = test_mask.rs_and(fs)
    test_data = data.mask(test_mask)
    
    corr_arr = np.array(list(u.bootstrap_list(np.array(td), np.mean, n=n_boots)
                             for td in test_data[correct_field]))

    train_mask = train_mask.rs_and(data[outcome_field] == 'correct')
    c1_mask_train = train_mask.rs_and(data[category_field] == 1)
    c2_mask_train = train_mask.rs_and(data[category_field] == 2)
    c1_mask_test = test_mask.rs_and(data[category_field] == 1)
    c2_mask_test = test_mask.rs_and(data[category_field] == 2)

    params = dict(skl_axes=True, time_zero_field='sampleOn')
    c1_train, xs = data.mask(c1_mask_train).get_populations(binsize, tbeg,
                                                            tend, binstep,
                                                            **params)
    c2_train = data.mask(c2_mask_train).get_populations(binsize, tbeg, tend,
                                                       binstep, **params)[0]
    c1_test_d = data.mask(c1_mask_test)
    c1_test = c1_test_d.get_populations(binsize, tbeg, tend, binstep,
                                        **params)[0]
    c1_test_outcomes = c1_test_d[correct_field]
    
    c2_test_d = data.mask(c2_mask_test)
    c2_test = c2_test_d.get_populations(binsize, tbeg, tend, binstep,
                                        **params)[0]
    c2_test_outcomes = c2_test_d[correct_field]
    if equal_trials:
        c1nt = data.mask(c1_mask_train).get_ntrls()
        c2nt = data.mask(c2_mask_train).get_ntrls()
        n_trls = np.min(np.stack((c1nt, c2nt), axis=1), axis=1)
        mask = n_trls >= min_trls
        new_min = np.min(n_trls[mask])
        print(min_trls, new_min)
        print(n_trls)
        mask = n_trls >= new_min
        c1_train = np.array(c1_train, dtype=object)[mask]
        c2_train = np.array(c2_train, dtype=object)[mask]
        c1_test = np.array(c1_test, dtype=object)[mask]
        c2_test = np.array(c2_test, dtype=object)[mask]
        c1_test_outcomes = np.array(c1_test_outcomes, dtype=object)[mask]
        c2_test_outcomes = np.array(c2_test_outcomes, dtype=object)[mask]
        corr_arr = corr_arr[mask]
        if other_info_fields is not None:
            other_info = other_info[mask]
    else:
        resamples = 1
    outs_gen = np.zeros((len(c1_train), resamples, len(xs)))
    outs_train = np.zeros((len(c1_train), resamples, len(xs)))
    outcome_matching = []
    for i, c1_tr in enumerate(c1_train):
        for k in range(resamples):
            if equal_trials:
                c1_inds = np.random.choice(c1_tr.shape[2], min_trls,
                                           replace=False)
                c2_inds = np.random.choice(c2_train[i].shape[2], min_trls,
                                           replace=False)
                c1_tr = c1_tr[:, :, c1_inds]
                c2_tr = c2_train[i][:, :, c2_inds]
            else:
                c2_tr = c2_train[i]
            outs_train[i, k] = na.fold_skl(c1_tr, c2_tr, folds_n, mean=True,
                                           class_weight='balanced', **kwargs)
            out = na.decode_skl(c1_tr, c1_test[i], c2_tr, c2_test[i],
                                return_dists=True, **kwargs)
            outs_gen[i, k], bv1, bv2 = out
            gencat = np.concatenate((bv1, bv2), axis=1)
            if k == 0:
                gencats = np.zeros((resamples,) + gencat.shape)
            gencats[k] = gencat
        c1o = c1_test_outcomes[i]
        c2o = c2_test_outcomes[i]
        resp = np.concatenate((c1o, c2o))
        cat = np.concatenate((np.zeros(len(c1o)),
                              np.ones(len(c2o))))
        outcome_matching.append((resp, cat, gencats))
    out = (outs_train, outs_gen, xs, corr_arr, outcome_matching,
           other_info)
    return out
