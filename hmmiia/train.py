import pdb
import time
import itertools
import os
import shutil
import sys

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from jax import value_and_grad, jit
from jax import lax, ops
from jax.experimental import optimizers

from sica_func_estimators import init_coder_params
from hmm_functions import mbatch_emission_likelihood, emission_likelihood
from hmm_functions import mbatch_fwd_bwd_algo, mbatch_m_step
from hmm_functions import forward_backward_algo, viterbi_algo
from utils import matching_sources_corr, clustering_acc
from itcl.itcl_train import train as itcl_train
from subfunc.showdata import *

# train HM-nICA
def train(data_dict, train_dict, seed_dict, tcl_dict, results_dict, path=None):
    """Train HM-NLICA model using a minibatch implementation of the algorithm
    described in the paper.

    Args:
        data_dict (dict.): dictionary of required data in the form of:
            {'x_data': observed signals (array),
             's_data': true latent component, for evaluation (array),
             'state_seq': true latent state sequece (array)}.
        train_dict (dict.): dictionary of variables related to optimization
            of form:
                {'mix_depth': num. layers in mixing/estimator MLP (int), for
                    example mix_depth=1 is linear ICA,
                 'hidden_size': num. hidden units per MLP layer (int),
                 'learning_rate': step size for optimizer (float),
                 'num_epochs': num. training epochs (int),
                 'subseq_len': length of time sequences in a minibatch (int),
                 'minib_size': num. sub-sequences in a minibatch (int),
                 'decay_rate': multiplier for decaying learning rate (float),
                 'decay_steps': num. epochs per which to decay lr (int)}.
        seed_dict (dict.): dictionary of seeds for reproducible stochasticity
            of form:
                {'est_mlp_seed': seed to initialize MLP parameters (int),
                 'est_distrib_seed': seed to initialize exp fam params (int)}.
        results_dict (dict.): stores data to save (see main.py).

    Returns:
        s_est (array): estimated independent components.
        sort_idx (array): best matching indices of components to true indices.
        results_dict (dict): to save all evaluation and training results.
        est_params (list): list of all estimated parameter arrays.
    """
    # unpack data
    x = data_dict['x_data']
    s_true = data_dict['s_data']
    state_seq = data_dict['state_seq']

    # set data dimensions
    N = x.shape[1]
    T = x.shape[0]
    # note we have to equalize for possible combinations that sica can do
    K = np.unique(state_seq, axis=0).shape[0]

    # unpack training variables
    mix_depth = train_dict['mix_depth']
    hidden_size = train_dict['hidden_size']
    learning_rate = train_dict['learning_rate']
    num_epochs = train_dict['num_epochs']
    subseq_len = train_dict['subseq_len']
    minib_size = train_dict['minib_size']
    decay_rate = train_dict['decay_rate']
    decay_steps = train_dict['decay_steps']

    print("Training with N={n}, T={t}, K={k}\t"
          "mix_depth={md}".format(n=N, t=T, k=K, md=mix_depth))

    # concatenate xt and xtm-1
    xttm1 = np.concatenate([x[1:,:], x[:-1,:]], axis=1)
    T = xttm1.shape[0]
    s_true = s_true[:T, :]
    state_seq = state_seq[:T]

    # initialize parameters for mlp function approximator
    key = jrandom.PRNGKey(seed_dict['est_mlp_seed'])
    mlp_params = init_coder_params(2*N, N, hidden_size,
                                   mix_depth-1, key)

    # trackers for training
    tcl_loss = []
    tcl_accu = []

#    # Initialize MLP by TCL
#    if path is not None:
#        train_dir = path.replace('.pkl','')
#
#        if train_dir.find("/storage/") > -1:
#            if os.path.exists(train_dir):
#                print("delete savefolder: {0:s}...".format(train_dir))
#                shutil.rmtree(train_dir)  # Remove folder
#            print("make savefolder: {0:s}...".format(train_dir))
#            os.makedirs(train_dir)  # Make folder
#        else:
#            assert False, "savefolder looks wrong"
#
#    num_segmentdata = 2**5
#    num_segment = int(np.ceil(T / num_segmentdata))
#    y = np.tile(np.arange(num_segment),[num_segmentdata,1]).T.reshape([-1])[:T]
#    mlp_init, mlp_init_ema, tcl_loss, tcl_accu = itcl_train(xttm1.copy(),
#               y.reshape(-1),
#               list_hidden_nodes=layer_sizes[1:],
#               list_hidden_nodes_z=layer_sizes[1:],
#               num_segment=num_segment,
#               initial_learning_rate=tcl_dict['initial_learning_rate'],
#               momentum=tcl_dict['momentum'],
#               max_steps=tcl_dict['max_steps'],
#               decay_steps=tcl_dict['decay_steps'],
#               decay_factor=tcl_dict['decay_factor'],
#               batch_size=tcl_dict['batch_size'],
#               train_dir=tcl_dict['train_dir'],
#               weight_decay=tcl_dict['weight_decay'],
#               checkpoint_steps=tcl_dict['checkpoint_steps'],
#               moving_average_decay=tcl_dict['moving_average_decay'],
#               summary_steps=tcl_dict['summary_steps'],
#               random_seed=tcl_dict['random_seed'])
#
#    for ln in range(len(mlp_params)):
#        # mlp_params[ln] = (jnp.array(mlp_init['layer.%d.weight' % ln]).astype(np.float64),
#        #                   jnp.array(mlp_init['layer.%d.bias' % ln]).astype(np.float64))
#        # mlp_params[ln] = (jnp.array(mlp_init_ema['layer.%d.weight' % ln]).astype(np.float64),
#        #                   jnp.array(mlp_init_ema['layer.%d.bias' % ln]).astype(np.float64))
#        mlp_params[ln] = (jnp.array(mlp_init_ema['layer.%d.weight' % ln]).astype(np.float64).T,
#                          jnp.array(mlp_init_ema['layer.%d.bias' % ln]).astype(np.float64))

    # initialize parameters for estimating distribution parameters
    np.random.seed(seed_dict['est_distrib_seed'])
    mu_est = np.random.uniform(-1., 1., size=(K, N))
    var_est = np.random.uniform(0.01, 1., size=(K, N))
    D_est = np.zeros(shape=(K, N, N))
    for k in range(K):
        D_est[k] = np.diag(var_est[k])

    # initialize transition parameter estimates
    A_est = np.eye(K) + 0.05
    A_est = A_est / A_est.sum(1, keepdims=True)
    pi_est = A_est.sum(0)/A_est.sum()

    # set up optimizer
    schedule = optimizers.exponential_decay(learning_rate,
                                            decay_steps=decay_steps,
                                            decay_rate=decay_rate)
    opt_init, opt_update, get_params = optimizers.adam(schedule)

    # set up loss function and training step
    @jit
    def calc_loss(params, input_data, marginal_posteriors,
                  mu_est, D_est, num_subseqs):
        """Calculates the loss for gradient M-step for function estimator.
        """
        lp_s, lp_x0, lp_J, _ = mbatch_emission_likelihood(params, input_data,
                                                          mu_est, D_est)
        expected_lp_x = jnp.sum(marginal_posteriors*lp_s,
                                -1) + lp_J + lp_x0[:, None]/lp_J.shape[1]
        # note correction for bias below
        return -expected_lp_x.mean()

    @jit
    def training_step(iter_num, input_data, marginal_posteriors,
                      mu_est, D_est, opt_state, num_subseqs):
        """Performs gradient m-step on the function estimator
               MLP parameters.
        """
        params = get_params(opt_state)
        loss, g = value_and_grad(calc_loss, argnums=0)(
            params, input_data,
            lax.stop_gradient(marginal_posteriors),
            mu_est, D_est, num_subseqs
        )
        return loss, opt_update(iter_num, g, opt_state)

    # function to load subsequence data for minibatches
    @jit
    def get_subseq_data(orig_data, subseq_array_to_fill):
        """Collects all sub-sequences into an array.
        """
        subseq_data = subseq_array_to_fill
        num_subseqs = subseq_data.shape[0]
        subseq_len = subseq_data.shape[1]

        def body_fun(i, subseq_data):
            """Function to loop over.
            """
            subseq_i = lax.dynamic_slice_in_dim(orig_data, i, subseq_len)
            subseq_data = ops.index_update(subseq_data, ops.index[i, :, :],
                                           subseq_i)
            return subseq_data
        return lax.fori_loop(0, num_subseqs, body_fun, subseq_data)

    # set up minibatch training
    num_subseqs = T-subseq_len+1
    assert num_subseqs >= minib_size
    num_full_minibs, remainder = divmod(num_subseqs, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    sub_data_holder = jnp.zeros((num_subseqs, subseq_len, 2*N))
    sub_data = get_subseq_data(xttm1, sub_data_holder)
    print("T: {t}\t"
          "subseq_len: {slen}\t"
          "minibatch size: {mbs}\t"
          "num minibatches: {nbs}".format(
              t=T, slen=subseq_len, mbs=minib_size, nbs=num_minibs))

    # initialize and train
    best_logl = -np.inf
    itercount = itertools.count()
    opt_state = opt_init(mlp_params)
    all_subseqs_idx = np.arange(num_subseqs)

    # evaluate the initialized model
    params_latest = get_params(opt_state)
    logp_s_all, logp_x0_all, logJ_all, s_est_all = emission_likelihood(
        params_latest, xttm1, mu_est, D_est)
    _, _, scalers = forward_backward_algo(logp_s_all, A_est, pi_est)
    logl_init = np.log(scalers).sum() + logJ_all.sum() + logp_x0_all
    # viterbi to estimate state prediction
    #est_seq = viterbi_algo(logp_s_all, A_est, pi_est)
    #cluster_acc_init = clustering_acc(np.array(est_seq), np.array(state_seq))
    # evaluate correlation of estimated and true independent components
    corr_diag, mean_abs_corr, s_est_sorted, sort_idx = matching_sources_corr(
        np.array(s_est_all), np.array(s_true))

    results_dict['init'] = {'logl': logl_init,
                            'corr': mean_abs_corr,
                            'corrdiag': corr_diag,
                            'tcl_loss': tcl_loss,
                            'tcl_accu': tcl_accu}

    for epoch in range(num_epochs):
        tic = time.time()
        # shuffle subseqs for added stochasticity
        np.random.shuffle(all_subseqs_idx)
        sub_data = sub_data.copy()[all_subseqs_idx]
        # train over minibatches
        for batch in range(num_minibs):
            if batch % 16 == 0:
                sys.stdout.write('\rMini-batch... %d/%d' % (batch + 1, num_minibs))
                sys.stdout.flush()

            # select sub-sequence for current minibatch
            batch_data = sub_data[batch*minib_size:(batch+1)*minib_size]

            # calculate emission likelihood using most recent parameters
            params = get_params(opt_state)
            logp_s, logp_x0, lpj, s_est = mbatch_emission_likelihood(
                params, batch_data, mu_est, D_est
            )

            # forward-backward algorithm
            marg_posteriors, pw_posteriors, scalers = mbatch_fwd_bwd_algo(
                logp_s, A_est, pi_est
            )

            # exact M-step for mean and variance
            mu_est, D_est, A_est, pi_est = mbatch_m_step(s_est,
                                                         marg_posteriors,
                                                         pw_posteriors)

            # SGD for mlp parameters
            loss, opt_state = training_step(next(itercount), batch_data,
                                            marg_posteriors, mu_est, D_est,
                                            opt_state, num_subseqs)

            assert not np.isnan(loss)

        sys.stdout.write('\r\n')

        if epoch % 1 == 0 or epoch == num_epochs - 1:
            # gather full data after each epoch for evaluation
            params_latest = get_params(opt_state)
            logp_s_all, logp_x0_all, logJ_all, s_est_all = emission_likelihood(
                   params_latest, xttm1, mu_est, D_est
            )
            _, _, scalers = forward_backward_algo(
                    logp_s_all, A_est, pi_est
                )
            logl_all = np.log(scalers).sum() + logJ_all.sum() + logp_x0_all

            # viterbi to estimate state prediction
            #est_seq = viterbi_algo(logp_s_all, A_est, pi_est)
            #cluster_acc = clustering_acc(np.array(est_seq), np.array(state_seq))

            # evaluate correlation of estimated and true independent components
            corr_diag, mean_abs_corr, s_est_sorted, sort_idx = matching_sources_corr(
                np.array(s_est_all), np.array(s_true)
            )

            # save results
            if logl_all > best_logl:
                best_epoch = epoch
                best_logl = logl_all
                best_logl_corr = mean_abs_corr
                best_logl_corrdiag = corr_diag
                #best_logl_acc = cluster_acc

            results_dict['results'].append({'epoch': epoch,
                                            'logl': logl_all,
                                            'corr': mean_abs_corr,
                                            'corrdiag': corr_diag})
            # print them
            print("Epoch: [{0}/{1}]\t"
                  "LogL: {logl:.2f}\t"
                  "mean corr between s and s_est {corr:.2f}\t"
                  "elapsed {time:.2f}".format(
                      epoch, num_epochs, logl=logl_all, corr=mean_abs_corr,
                      time=time.time()-tic))

    # pack data into tuples
    results_dict['best'] = {'epoch': best_epoch,
                            'logl': best_logl,
                            'corr': best_logl_corr,
                            'corrdiag': best_logl_corrdiag}
    # results_dict['results'].append({'best_logl': best_logl,
    #                                 'best_logl_corr': best_logl_corr,
    #                                 'best_logl_acc': best_logl_acc})
    est_params = (mu_est, D_est, A_est, est_seq)
    return s_est, sort_idx, results_dict, est_params
