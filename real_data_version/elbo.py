import pdb

import jax
import jax.numpy as jnp
import jax.random as jrandom
from fixed_points import *
from jax import vmap, jit
from jax.lax import scan
from jax.scipy.stats import multivariate_normal as gaussian
from jax.experimental import host_callback

from data_generation import gen_slds_linear_ica
from inference import make_inference, inference_fixedpoint
from func_estimators import encoder_mlp, decoder_mlp
from utils import get_prior_natparams, get_hmm_natparams
from utils import get_likelihood_natparams, get_transition_natparams
from utils import get_rhos
from utils import invmp, gaussian_sample_from_mu_prec

from functools import partial

import matplotlib.pyplot as plt

from jax.config import config
#config.update("jax_debug_nans", True)


#@jit
def sample_fwd_step(z_prev, pw_posterior_with_key):
    qzlag_z, key = pw_posterior_with_key
    mu_pw, prec_pw = qzlag_z
    d = z_prev.shape[0]
    mu_cond = mu_pw[d:]-invmp(prec_pw[d:, d:],
                              prec_pw[d:, :d]@(z_prev-mu_pw[:d]))
    prec_cond = prec_pw[d:, d:]
    z_new = gaussian_sample_from_mu_prec(mu_cond, prec_cond, key)
    return z_new, z_new


#@jit
def sample_forward(lds_posteriors, key):
    qz, qzlag_z = lds_posteriors
    mu, prec = qz
    keys = jrandom.split(key, mu.shape[0])
    z_prior = gaussian_sample_from_mu_prec(mu[0], prec[0], keys[0])
    z = scan(sample_fwd_step, z_prior, (qzlag_z, keys[1:]))[1]
    return jnp.vstack((z_prior, z))


#@jit
def gaussian_entropy(prec):
    d = prec.shape[0]
    V = invmp(prec, jnp.eye(prec.shape[0]))
    return 0.5*(jnp.linalg.slogdet(V)[1]+d*jnp.log(2*jnp.pi*jnp.e))


#@jit
def lds_entropy(qz_prec, qzlag_z_prec):
    marginal_entropies = vmap(lambda a:
                              vmap(gaussian_entropy)(a[1:-1]))(qz_prec)
    pw_entropies = vmap(vmap(gaussian_entropy))(qzlag_z_prec)
    return jnp.sum(pw_entropies)-jnp.sum(marginal_entropies)


#@jit
def E_logp_lds(prior_natparams, transition_natparams, qz, qzlag_z, qu):
    rho = vmap(get_rhos)((prior_natparams, transition_natparams),
                         (qz, qzlag_z))
    E_logp_z = jnp.sum(rho*jnp.exp(qu))
    return E_logp_z


#@jit
def KL_qp_u(hmm_natparams, qu, quu):
    pw_part = jnp.sum((quu-hmm_natparams[1][:, None, :, :])*jnp.exp(quu))
    marg_part = jnp.sum(jnp.exp(qu[:, 1:-1, :])*qu[:, 1:-1, :])+jnp.sum(
        jnp.exp(qu[:, 0, :])*hmm_natparams[0])
    return pw_part-marg_part


#@jit
def E_sampling_likelihood(x, s_sample, theta, R):
    x_sample_mu = vmap(vmap(lambda a: decoder_mlp(theta, a), in_axes=-1,
                       out_axes=-1))(s_sample)
    V = invmp(R, jnp.eye(R.shape[0]))
    E_logp_xs = vmap(lambda a: vmap(gaussian.logpdf, in_axes=(-1, -1, None),
                     out_axes=-1)(x, a, V))(x_sample_mu)
    return E_logp_xs.sum(1).mean()


#@partial(jit, static_argnums=(7, 8,))
def ELBO(x, R, lds_params, log_hmm_params, phi, theta, key,
         inference_iters, num_samples):
    # transform into natural parameter form
    M, T = x.shape
    N, K = log_hmm_params[0].shape
    d = lds_params[0].shape[-1]
    hmm_params = jax.tree_map(lambda a: jnp.exp(a), log_hmm_params)
    # ensure probabilities are normalized
    hmm_params = jax.tree_map(lambda a: a/jnp.sum(a, -1, keepdims=True),
                              hmm_params)
    # transform into natparams
    hmm_natparams = scan(get_hmm_natparams, None, hmm_params)[1]
    prior_natparams = vmap(lambda a: scan(get_prior_natparams,
                                          None, a)[1])(lds_params[:2])
    transition_natparams = vmap(lambda a: scan(get_transition_natparams,
                                None, a)[1])(lds_params[2:])
    likelihood_natparams = vmap(lambda a: encoder_mlp(phi, a),
                                in_axes=-1, out_axes=(-1, -1))(x)

    # initialize posteriors
    qu = jnp.log(jnp.ones((N, T, K))/K)
    quu = jnp.log(jnp.ones((N, T-1, K, K)) / (K**2))
    qz = (jnp.zeros((N, T, d)),
          jnp.tile(jnp.eye(d)[None, None], (N, T, 1, 1)))
    qzlag_z = (jnp.zeros((N, T-1, 2*d)),
               jnp.tile(jnp.eye(2*d)[None, None], (N, T-1, 1, 1)))

    # run inference
    inference_runner = make_inference(hmm_natparams, prior_natparams,
                                      transition_natparams,
                                      likelihood_natparams)
    qz, qzlag_z, qu, quu = scan(inference_runner, (qz, qzlag_z, qu, quu),
                                jnp.arange(inference_iters))[0]

    # sample
    z_sample = vmap(
        lambda k: vmap(sample_forward)((qz, qzlag_z), jrandom.split(k, N))
    )(jrandom.split(key, num_samples))
    s_sample = z_sample[:, :, :, 0]

    # compute elbo
    E_logp_xs = E_sampling_likelihood(x, s_sample, theta, R)
    E_logp_z = E_logp_lds(prior_natparams, transition_natparams,
                          qz, qzlag_z, qu)
    H_z = lds_entropy(qz[1], qzlag_z[1])
    KL_u = KL_qp_u(hmm_natparams, qu, quu)
    elbo = E_logp_xs+E_logp_z+H_z-KL_u
    #host_callback.id_print({'E_logp_xs': E_logp_xs, 'E_logp_z': E_logp_z,
    #                        'H_z': H_z, '-KL_u': -KL_u})
    return elbo, (qz, qzlag_z, qu, quu)


#@partial(jit, static_argnums=(7, 8,))
def avg_neg_ELBO(x, mix_params, lds_params, hmm_params, phi, theta,
                 key, inference_iters, num_samples):
    keys = jrandom.split(key, x.shape[0])
    elbo, posteriors = vmap(
        lambda a, b: ELBO(a, mix_params, lds_params, hmm_params, phi, theta,
                          b, inference_iters, num_samples)
    )(x, keys)
    return -elbo.mean()/x.shape[-1], posteriors


def vmp_ELBO(x, R, lds_params, log_hmm_params, phi, theta,
             R_q, lds_params_q, log_hmm_params_q,
             key, inference_iters, num_samples):
    # transform into natural parameter form
    M, T = x.shape
    N, K = log_hmm_params[0].shape
    d = lds_params[0].shape[-1]
    hmm_params = jax.tree_map(lambda a: jnp.exp(a), log_hmm_params)
    hmm_params_q = jax.tree_map(lambda a: jnp.exp(a), log_hmm_params_q)
    # ensure probabilities are normalized
    hmm_params = jax.tree_map(lambda a: a/jnp.sum(a, -1, keepdims=True),
                              hmm_params)
    hmm_params_q = jax.tree_map(lambda a: a/jnp.sum(a, -1, keepdims=True),
                                hmm_params_q)

    # transform into natparams
    hmm_natparams = scan(get_hmm_natparams, None, hmm_params)[1]
    prior_natparams = vmap(lambda a: scan(get_prior_natparams,
                                          None, a)[1])(lds_params[:2])
    transition_natparams = vmap(lambda a: scan(get_transition_natparams,
                                None, a)[1])(lds_params[2:])
    likelihood_natparams = vmap(lambda a: encoder_mlp(phi, a),
                                in_axes=-1, out_axes=(-1, -1))(x)
    # similarly for the vmp auxiliary params
    hmm_natparams_q = scan(get_hmm_natparams, None, hmm_params_q)[1]
    prior_natparams_q = vmap(lambda a: scan(get_prior_natparams,
                             None, a)[1])(lds_params_q[:2])
    transition_natparams_q = vmap(lambda a: scan(get_transition_natparams,
                                  None, a)[1])(lds_params_q[2:])

    # initialize posteriors
    qu = jnp.log(jnp.ones((N, T, K))/K)
    quu = jnp.log(jnp.ones((N, T-1, K, K)) / (K**2))
    qz = (jnp.zeros((N, T, d)),
          jnp.tile(jnp.eye(d)[None, None], (N, T, 1, 1)))
    qzlag_z = (jnp.zeros((N, T-1, 2*d)),
               jnp.tile(jnp.eye(2*d)[None, None], (N, T-1, 1, 1)))

    # run inference
    inference_runner = make_inference(hmm_natparams_q, prior_natparams_q,
                                      transition_natparams_q,
                                      likelihood_natparams)
    qz, qzlag_z, qu, quu = scan(inference_runner, (qz, qzlag_z, qu, quu),
                                jnp.arange(inference_iters))[0]

    # sample
    z_sample = vmap(
        lambda k: vmap(sample_forward)((qz, qzlag_z), jrandom.split(k, N))
    )(jrandom.split(key, num_samples))
    s_sample = z_sample[:, :, :, 0]

    # compute elbo
    E_logp_xs = E_sampling_likelihood(x, s_sample, theta, R)
    E_logp_z = E_logp_lds(prior_natparams, transition_natparams,
                          qz, qzlag_z, qu)
    H_z = lds_entropy(qz[1], qzlag_z[1])
    KL_u = KL_qp_u(hmm_natparams, qu, quu)
    elbo = E_logp_xs+E_logp_z+H_z-KL_u
    #host_callback.id_print({'E_logp_xs': E_logp_xs, 'E_logp_z': E_logp_z,
    #                        'H_z': H_z, '-KL_u': -KL_u})
    return elbo, (qz, qzlag_z, qu, quu)


#@partial(jit, static_argnums=(7, 8,))
def vmp_avg_neg_ELBO(x, R, lds_params, hmm_params, phi, theta,
                     R_q, lds_params_q, log_hmm_params_q,
                     key, inference_iters, num_samples):
    keys = jrandom.split(key, x.shape[0])
    elbo, posteriors = vmap(
        lambda a, b: vmp_ELBO(a, R, lds_params, hmm_params, phi, theta,
                              R_q, lds_params_q, log_hmm_params_q,
                              b, inference_iters, num_samples)
    )(x, keys)
    return -elbo.mean()/x.shape[-1], posteriors


if __name__ == "__main__":
    # generate data
    key = jrandom.PRNGKey(3)
    N = 4
    T = 100000
    d = 3
    K = 2
    stay_prob = 0.99

    # create data
    key, datakey = jrandom.split(key)
    x, z, z_mu, states, *params = gen_slds_linear_ica(N, T, K, d,
                                                      stay_prob, datakey)
    mix_params, lds_params, hmm_params = params


    # test elbo
    ELBO(x, mix_params, lds_params, hmm_params)

    pdb.set_trace()
