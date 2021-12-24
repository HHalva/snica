import os
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/appl/opt/CUDA/10.2.89-GCC-8.3.0"
#os.environ["MPLCONFIGDIR"]  = "/wrk/users/herhal"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import itertools
import pickle
import numpy as onp

import pdb

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, value_and_grad, lax
from jax.lax import scan
from jax.ops import index, index_add, index_update
from jax.experimental import optimizers, host_callback

from data_generation import generate_slds
from elbo import avg_neg_ELBO, vmp_avg_neg_ELBO
from func_estimators import init_encoder_params, init_decoder_params
from utils import matching_sources_corr, tree_get_idx, plot_ic
from utils import nsym_grad, sym_grad
from functools import partial
# from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from jax.config import config
#config.update("jax_debug_nans", True)


def train(x, args):
    print("running with following:", args)
    # dimensions
    M, T = x.shape
    N = args.n
    d = args.d
    K = args.k
    print('M:', M, 'N:', N, 'T:', T, 'd:', d, 'K:', K)
    # training args
    p_stay = args.prob_stay
    num_epochs = args.num_epochs
    decay_interval = args.decay_interval
    decay_rate = args.decay_rate
    subseq_len = args.subseq_len
    minib_size = args.minib_size
    # inference routine args
    inference_iters = args.inference_iters
    num_samples = args.num_samples
    vmp = args.vmp
    # function estimators
    enc_hidden_units = args.hidden_units_enc
    dec_hidden_units = args.hidden_units_dec
    enc_hidden_layers = args.hidden_layers_enc
    dec_hidden_layers = args.hidden_layers_dec
    lr_nn = args.nn_learning_rate
    lr_gm = args.gm_learning_rate
    # printing and saving frequency
    save_freq = args.save_freq
    print_freq = args.print_freq
    # set-up random key
    est_key = jrandom.PRNGKey(args.est_seed)

    # initialize graphical model parameters
    est_key, *gp_keys = jrandom.split(est_key, N+1)
    # initialize noise level for x
    R_est = jnp.eye(M)*jrandom.uniform(
        gp_keys[0], shape=(M,), minval=1., maxval=10.)*args.init_xprec_scale

    # initialize hmm params
    pi = jnp.ones((N, K))/K
    nondiag_prob = (1-p_stay)/(K-1)
    A = jnp.zeros((K, K))+nondiag_prob
    A = index_update(A, jnp.diag_indices_from(A), p_stay)
    A = jnp.tile(A[None], (N, 1, 1))
    assert (jnp.sum(A, -1, keepdims=True) == jnp.ones((N, K, 1))).all()
    hmm_est = (jnp.log(pi), jnp.log(A))
    # initialize lds params (need to rethink this still)
    _, lds_est = vmap(lambda a: generate_slds(K, a))(jnp.array(gp_keys))
    # re-order (fix later in data generation code) and pack
    B_est, b_est, b_prior_est, _, _, Q_est, Q_prior_est, _ = lds_est
    lds_est = (b_prior_est, Q_prior_est, B_est, b_est, Q_est)

    # initialize func estimators
    key, enc_key, dec_key = jrandom.split(est_key, 3)
    phi = init_encoder_params(M, N, enc_hidden_units,
                              enc_hidden_layers, enc_key)
    theta = init_decoder_params(M, N, dec_hidden_units,
                                dec_hidden_layers, dec_key)

    # initialize training
    nn_schedule = optimizers.exponential_decay(lr_nn, decay_interval,
                                               decay_rate)
    gm_schedule = optimizers.exponential_decay(lr_gm, decay_interval,
                                               decay_rate)
    opt_init_nn, opt_update_nn, get_params_nn = optimizers.adam(nn_schedule)
    opt_init_gm, opt_update_gm, get_params_gm = optimizers.sgd(gm_schedule)
    opt_state_nn = opt_init_nn((phi, theta))
    opt_state_gm = opt_init_gm((R_est, lds_est, hmm_est))

    #####################################
    # add option to return to checkpoint#
    #####################################
    # return to converged enc/dec
    #encdec_ckpt = pickle.load(
    #    open(os.path.join("/home/local/herhal/Documents/sica",
    #                      "encdec_ckpt.pkl"), "rb")
    #)
    #opt_state_nn = optimizers.pack_optimizer_state(encdec_ckpt)

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
    def training_step(epoch_num, opt_state_nn, opt_state_gm, x,
                      inference_iters, num_samples, key):
        """Performs gradient step on the function estimator
               MLP parameters on the ELBO.
        """
        key, subkey = jrandom.split(key)
        phi, theta = get_params_nn(opt_state_nn)
        R_est, lds_est, hmm_est = get_params_gm(opt_state_gm)
        # get gradients
        (n_elbo, posteriors), g = value_and_grad(
            avg_neg_ELBO, argnums=(1, 2, 3, 4, 5,), has_aux=True)(
                x, R_est, lds_est, hmm_est, phi, theta,
                subkey, inference_iters, num_samples
        )

        # overall grad adjustment for all variables
        g = jax.tree_map(lambda a: a*num_subseqs, g)
        R_g, lds_g, hmm_g, phi_g, theta_g = g
        b_prior_g, Q_prior_g, B_g, b_g, Q_g = lds_g
        pi_g, A_g = hmm_g
        # adjust grads
        R_g, phi_g, theta_g = jax.tree_map(lambda a: a/subseq_len,
                                           (R_g, phi_g, theta_g))
        B_g, b_g, Q_g, A_g = jax.tree_map(lambda a: a/(subseq_len-1),
                                          (B_g, b_g, Q_g, A_g))
        # symmetrization of precision matrix grads - can also use sym_grad()
        R_g = sym_grad(R_g)
        Q_prior_g = vmap(vmap(sym_grad))(Q_prior_g)
        Q_g = vmap(vmap(sym_grad))(Q_g)

        # pack up
        lds_g = (b_prior_g, Q_prior_g, B_g, b_g, Q_g)
        hmm_g = (pi_g, A_g)

        # perform gradient updates
        opt_state_nn = opt_update_nn(epoch_num, (phi_g, theta_g),
                                     opt_state_nn)
        opt_state_gm = opt_update_gm(epoch_num, (R_g, lds_g, hmm_g),
                                     opt_state_gm)
        return n_elbo, posteriors, (opt_state_nn, opt_state_gm)

    @partial(jit, static_argnums=(4, 5))
    def vmp_training_step(epoch_num, opt_state_nn, opt_state_gm, x,
                          inference_iters, num_samples, key):
        """Performs gradient step on the function estimator
               MLP parameters on the ELBO.
        """
        key, subkey = jrandom.split(key)
        phi, theta = get_params_nn(opt_state_nn)
        R_est, lds_est, hmm_est = get_params_gm(opt_state_gm)
        # get gradients
        R_q = R_est
        lds_q = lds_est
        hmm_q = hmm_est
        (n_elbo, posteriors), g = value_and_grad(
            vmp_avg_neg_ELBO, argnums=(1, 2, 3, 4, 5,), has_aux=True)(
                x, R_est, lds_est, hmm_est, phi, theta,
                R_q, lds_q, hmm_q,
                subkey, inference_iters, num_samples
        )

        # overall grad adjustment for all variables
        g = jax.tree_map(lambda a: a*num_subseqs, g)
        R_g, lds_g, hmm_g, phi_g, theta_g = g
        b_prior_g, Q_prior_g, B_g, b_g, Q_g = lds_g
        pi_g, A_g = hmm_g
        # adjust grads
        R_g, phi_g, theta_g = jax.tree_map(lambda a: a/subseq_len,
                                           (R_g, phi_g, theta_g))
        B_g, b_g, Q_g, A_g = jax.tree_map(lambda a: a/(subseq_len-1),
                                          (B_g, b_g, Q_g, A_g))
        # symmetrization of precision matrix grads - can also use sym_grad()
        R_g = sym_grad(R_g)
        Q_prior_g = vmap(vmap(sym_grad))(Q_prior_g)
        Q_g = vmap(vmap(sym_grad))(Q_g)

        # pack up
        lds_g = (b_prior_g, Q_prior_g, B_g, b_g, Q_g)
        hmm_g = (pi_g, A_g)

        # perform gradient updates
        opt_state_nn = opt_update_nn(epoch_num, (phi_g, theta_g),
                                     opt_state_nn)
        opt_state_gm = opt_update_gm(epoch_num, (R_g, lds_g, hmm_g),
                                     opt_state_gm)
        return n_elbo, posteriors, (opt_state_nn, opt_state_gm)

    # set plot
    fig, ax = plt.subplots(2, N, figsize=(10 * N, 6),
                           gridspec_kw={'height_ratios': [1, 2]})
    # train
    best_elbo = -jnp.inf
    itercount = itertools.count()
    shuffle_key = jrandom.PRNGKey(9999)
    for epoch in range(num_epochs):
        tic = time.time()
        niters = min(inference_iters, ((epoch // 100) + 1) * 5)
        shuffle_key, shuffkey = jrandom.split(shuffle_key)
        sub_data = jrandom.permutation(shuffkey, sub_data)
        # train over minibatches
        for it in range(num_minibs):
            # select sub-sequence for current minibatch
            x_it = sub_data[it*minib_size:(it+1)*minib_size]
            key, trainkey = jrandom.split(key, 2)

            # training step
            if vmp:
                n_elbo, posteriors, (opt_state_nn, opt_state_gm) = vmp_training_step(
                    next(itercount), opt_state_nn, opt_state_gm, x_it, niters,
                    num_samples, trainkey)
            else:
                n_elbo, posteriors, (opt_state_nn, opt_state_gm) = training_step(
                    next(itercount), opt_state_nn, opt_state_gm, x_it, niters,
                    num_samples, trainkey)

            print("*Epoch: [{0}/{1}]\t"
                  "minibatch: [{2}/{3}]\t"
                  "avg. ELBO over minibatch: {4}\t"
                  "num. infernce iters: {5}\t".format(epoch, num_epochs,
                                                      it, num_minibs,
                                                      -n_elbo, niters))
            if -n_elbo > best_elbo:
                best_elbo = -n_elbo
                best_opt_state_nn = opt_state_nn
                best_opt_state_gm = opt_state_gm
                if it % save_freq == 0:
                    encdec_ckpt = optimizers.unpack_optimizer_state(opt_state_nn)
                    gm_ckpt = optimizers.unpack_optimizer_state(opt_state_gm)
                    pickle.dump(encdec_ckpt, open(os.path.join(args.out_dir,
                                "encdec_ckpt.pkl"), "wb"))
                    pickle.dump(gm_ckpt, open(os.path.join(args.out_dir,
                                "gm_ckpt.pkl"), "wb"))

        print("Epoch took: ", time.time()-tic)
    return get_params_nn(best_opt_state_nn), get_params_gm(best_opt_state_gm), best_elbo
