import os
import time
import itertools
import pdb

from jax.config import config
config.update("jax_debug_nans", True)

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jax import vmap, jit, value_and_grad, lax
from jax.ops import index, index_update
from jax.lax import cond
from optax import chain, piecewise_constant_schedule, scale_by_schedule

from functools import partial
from elbo import avg_neg_ELBO
from func_estimators import init_encoder_params, init_decoder_params
from func_estimators import decoder_mlp

from utils import matching_sources_corr, plot_ic
from utils import nsym_grad, sym_grad, get_prec_mat
from utils import save_best, load_best_ckpt

import matplotlib.pyplot as plt


def full_train(x, f, z, z_mu, states, params, args, est_key):
    print("Running with:", args)
    # unpack some of the args
    N = z_mu.shape[0]
    M, T = x.shape
    num_epochs = args.num_epochs
    inference_iters = args.inference_iters
    num_samples = args.num_samples
    enc_hidden_units = args.hidden_units_enc
    dec_hidden_units = args.hidden_units_dec
    enc_hidden_layers = args.hidden_layers_enc
    dec_hidden_layers = args.hidden_layers_dec
    lr_nn = args.nn_learning_rate
    lr_gm = args.gm_learning_rate
    decay_interval = args.decay_interval
    decay_rate = args.decay_rate
    subseq_len = args.subseq_len
    minib_size = args.minib_size
    burnin_len = args.burnin
    plot_freq = args.plot_freq
    eval_freq = args.eval_freq
    mix_params, lds_params, hmm_params = params
    _, K, d = lds_params[0].shape

    # initialize pgm parameters randomly
    est_key = jrandom.PRNGKey(est_key)
    est_key, Rkey = jrandom.split(est_key)
    est_key, *hmmkeys = jrandom.split(est_key, 3)
    est_key, *ldskeys = jrandom.split(est_key, 6)
    x_vars = jnp.diag(jnp.cov(x))
    R_est = jnp.linalg.inv(jnp.diag(
        jrandom.uniform(Rkey, (M,), minval=0.1*jnp.min(x_vars),
                        maxval=0.5*jnp.max(x_vars))))
    hmm_est = jax.tree_map(
        lambda a: jnp.log(a / a.sum(-1, keepdims=True)),
        (jrandom.uniform(hmmkeys[0], (N, K)),
         jrandom.uniform(hmmkeys[1], (N, K, K)))
    )
    b_prior_est = jrandom.uniform(ldskeys[0], (N, K, d), minval=-1, maxval=1)
    b_est = jrandom.uniform(ldskeys[1], (N, K, d), minval=-1, maxval=1)
    B_est = jrandom.uniform(ldskeys[2], (N, K, d, d), minval=-1, maxval=1)
    Q_prior_est = vmap(lambda k: get_prec_mat(d, 10., k)*jnp.eye(d))(
        jrandom.split(ldskeys[3], N*K)).reshape((N, K, d, d))
    Q_est = vmap(lambda k: get_prec_mat(d, 10., k)*jnp.eye(d))(
        jrandom.split(ldskeys[4], N*K)).reshape((N, K, d, d))
    lds_est = (b_prior_est, Q_prior_est, B_est, b_est, Q_est)

    # for debugging at ground truth pgm parameters
    if args.gt_gm_params:
        R_est = mix_params[1]
        lds_est = lds_params
        hmm_est = jax.tree_map(lambda a: jnp.log(a), hmm_params)

    # initialize func estimators
    key, enc_key, dec_key = jrandom.split(est_key, 3)
    theta = init_decoder_params(M, N, dec_hidden_units,
                                dec_hidden_layers, dec_key)

    if args.l > 0:
        phi = init_encoder_params(M, N, enc_hidden_units,
                                  enc_hidden_layers, enc_key)
    if args.l == 0:
        # in linear case set bias to zero (also need to transpose matrix)
        theta = [(theta[0][0], jnp.zeros(theta[0][1].shape))]
        # also in linear case set phi=theta as we need phi to run the code but
        # phi variable is not actually used
        phi = theta

    # initialize training
    gm_params = (R_est, lds_est, hmm_est)
    nn_params = (phi, theta)
    all_params = (gm_params, nn_params)
    param_labels = ('gm', 'nn')
    schedule_fn = piecewise_constant_schedule(1., {decay_interval: decay_rate})
    tx = optax.multi_transform({
        'gm': chain(optax.adam(lr_gm), scale_by_schedule(schedule_fn)),
        'nn': chain(optax.adam(lr_nn), scale_by_schedule(schedule_fn))},
        param_labels)
    opt_state = tx.init(all_params)
    start_epoch = 0

    # option to resume to checkpoint
    if args.resume_best:
        start_epoch, all_params, opt_state, tx = load_best_ckpt(args)

    # functions to define sampling for SVI
    @jit
    def get_subseq_data(orig_data, subseq_array_to_fill):
        """Collects all sub-sequences into an array.
        """
        subseq_data = subseq_array_to_fill
        num_subseqs = subseq_data.shape[0]
        subseq_len = subseq_data.shape[-1]

        def body_fun(i, subseq_data):
            """Function to loop over.
            """
            subseq_i = lax.dynamic_slice_in_dim(orig_data, i,
                                                subseq_len, axis=1)
            subseq_data = index_update(subseq_data, index[i, :, :],
                                       subseq_i)
            return subseq_data
        return lax.fori_loop(0, num_subseqs, body_fun, subseq_data)

    # set up minibatch training
    num_subseqs = T-subseq_len+1
    assert num_subseqs >= minib_size
    num_full_minibs, remainder = divmod(num_subseqs, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    sub_data_holder = jnp.zeros((num_subseqs, M, subseq_len))
    sub_data = get_subseq_data(x, sub_data_holder)
    print("T: {t}\t"
          "subseq_len: {slen}\t"
          "minibatch size: {mbs}\t"
          "num minibatches: {nbs}".format(
              t=T, slen=subseq_len, mbs=minib_size, nbs=num_minibs))

    @partial(jit, static_argnums=(4, 5))
    def training_step(epoch_num, params, opt_state, x,
                      inference_iters, num_samples, burnin, key):
        """Performs gradient step on the function estimator
               MLP parameters on the ELBO.
        """
        # unpack
        key, subkey = jrandom.split(key)
        R_est, lds_est, hmm_est = params[0]
        phi, theta = params[1]

        # option to anneal elbo KL terms by factor
        nu = cond(burnin > 0,
                  lambda _: jnp.clip(epoch_num/(burnin+1e-5), a_max=1.0),
                  lambda _: 1., burnin)

        # get gradients
        (n_elbo, posteriors), g = value_and_grad(
            avg_neg_ELBO, argnums=(1, 2, 3, 4, 5,), has_aux=True)(
                x, R_est, lds_est, hmm_est, phi, theta, nu,
                subkey, inference_iters, num_samples, minibatch=True
        )

        # overall grad adjustment for all variables due to subsampling
        g = jax.tree_map(lambda a: a*num_subseqs, g)
        R_g, lds_g, hmm_g, phi_g, theta_g = g
        b_prior_g, Q_prior_g, B_g, b_g, Q_g = lds_g
        pi_g, A_g = hmm_g

        # specific gradient adjustments due to subsampling
        R_g, phi_g, theta_g = jax.tree_map(lambda a: a/subseq_len,
                                           (R_g, phi_g, theta_g))
        B_g, b_g, Q_g, A_g = jax.tree_map(lambda a: a/(subseq_len-1),
                                          (B_g, b_g, Q_g, A_g))
        b_prior_g, Q_prior_g, pi_g = jax.tree_map(lambda a: a/num_subseqs,
                                                  (b_prior_g, Q_prior_g, pi_g))

        # symmetrization of precision matrix grads - can also use nsym_grad()
        def sym_diag_grads(mat): return sym_grad(mat)*jnp.eye(mat.shape[0])

        R_g = sym_diag_grads(R_g)
        Q_prior_g = vmap(vmap(sym_diag_grads))(Q_prior_g)
        Q_g = vmap(vmap(sym_diag_grads))(Q_g)

        # pack up
        lds_g = (b_prior_g, Q_prior_g, B_g, b_g, Q_g)
        hmm_g = (pi_g, A_g)
        gm_g = (R_g, lds_g, hmm_g)
        nn_g = (phi_g, theta_g)
        g = (gm_g, nn_g)

        # perform gradient updates
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return n_elbo, posteriors, params, opt_state


    @partial(jit, static_argnums=(4, 5))
    def infer_step(epoch_num, params, opt_state, x,
                   inference_iters, num_samples, burnin, key):
        """Perform inference without gradient step for eval purposes
               MLP parameters on the ELBO.
        """
        # unpack
        key, subkey = jrandom.split(key)
        R_est, lds_est, hmm_est = params[0]
        phi, theta = params[1]

        # always turn annealing off for eval
        nu = 1.

        # inference step
        n_elbo, posteriors = avg_neg_ELBO(x, R_est, lds_est, hmm_est, phi,
                                          theta, nu, subkey, inference_iters,
                                          num_samples)
        return n_elbo, posteriors


    # set plot
    fig, ax = plt.subplots(2, N, figsize=(10 * N, 6),
                           gridspec_kw={'height_ratios': [1, 2]})
    ax2 = ax
    for n in range(N):
        ax2[1, n] = ax[1, n].twinx()

    # train
    best_elbo = -jnp.inf
    itercount = itertools.count()
    shuffle_key = jrandom.PRNGKey(9999)
    eval_key = jrandom.PRNGKey(9999999)
    for epoch in range(num_epochs):
        tic = time.time()
        #niters = min(inference_iters, ((epoch // 100) + 1) * 5)
        shuffle_key, shuffkey = jrandom.split(shuffle_key)
        sub_data = jrandom.permutation(shuffkey, sub_data)
        # train over minibatches
        for it in range(num_minibs):
            # adjust number of iterations
            niters = min(inference_iters, ((it // 100) + 1) * 5)
            # select sub-sequence for current minibatch
            x_it = sub_data[it*minib_size:(it+1)*minib_size]
            key, trainkey = jrandom.split(key, 2)

            if not args.eval_only:
                # training step on minibatch
                n_elbo, posteriors, all_params, opt_state = training_step(
                    epoch, all_params, opt_state, x_it, niters,
                    num_samples, burnin_len, trainkey)

            # evaluate on full data at chosen frequency
            if it % eval_freq == 0 or args.eval_only:
                # inference on full data
                n_elbo, posteriors = infer_step(epoch, all_params,
                                                opt_state, x, niters,
                                                num_samples, burnin_len,
                                                trainkey)
                # evaluate
                qz, qzlag_z, qu, quu = posteriors
                mcc, _, sort_idx = matching_sources_corr(qz[0][:, :, 0],
                                                         z_mu[:, :, 0])
                f_mu_est = vmap(decoder_mlp, in_axes=(None, -1),
                                out_axes=-1)(all_params[1][1], qz[0][:, :, 0])
                denoise_mcc = jnp.abs(jnp.diag(
                    jnp.corrcoef(f_mu_est, f)[:M, M:])).mean()

                print("*Epoch: [{0}/{1}]\t"
                      "Minibatch: [{2}/{3}]\t"
                      "ELBO: {4}\t"
                      "mcc: {corr: .2f}\t"
                      "denoise mcc: {dcorr: .2f}\t"
                      "num. infernce iters: {5}\t"
                      "eseed: {es}\t"
                      "pseed: {ps}".format(epoch, num_epochs, it, num_minibs,
                                           -n_elbo, niters, corr=mcc,
                                           dcorr=denoise_mcc, es=args.est_seed,
                                           ps=args.param_seed))

                if it % plot_freq == 0 or args.eval_only:
                    # plot
                    plot_start = int(T/2)
                    plot_len = 500
                    plot_end = plot_start+plot_len
                    for n in range(N):
                        qz_mu_n = qz[0][sort_idx][n][plot_start:plot_end]
                        qz_prec_n = qz[1][sort_idx][n][plot_start:plot_end]
                        qu_n = jnp.exp(qu[sort_idx][n][plot_start:plot_end])
                        u_n = states[n][plot_start:plot_end]
                        z_mu_n = z_mu[n][plot_start:plot_end]
                        plot_ic(u_n, z_mu_n, qu_n, qz_mu_n, qz_prec_n,
                                ax[0, n], ax[1, n], ax2[1, n])
                    plt.pause(.5)

                # saving
                if -n_elbo > best_elbo:
                    best_elbo = -n_elbo
                    best_params = all_params
                    best_posters = posteriors
                    save_best(epoch, args, all_params, opt_state, tx)

                if args.eval_only:
                    return best_params, best_posters, best_elbo
        print("Epoch took: ", time.time()-tic)
    return best_params, best_posters, best_elbo
