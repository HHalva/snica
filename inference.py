from jax.config import config
config.update("jax_enable_x64", True)

import pdb

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit
from jax.lax import scan
from jax.ops import index, index_update, index_add
from jax.scipy.special import logsumexp
from jax.experimental import host_callback

from utils import get_E_likelihood_natparams, get_gauss_params
from utils import get_resp_wgt_natparams, get_rhos
from utils import tree_sum, tree_droplast, tree_dropfirst
from utils import tree_prepend, tree_append, tree_get_idx
from utils import invmp

import matplotlib.pyplot as plt

from jax.config import config


#@jit
def hmm_fwd_pass(eta_in, eta_current):
    eta_A, eta_likelihood = eta_current
    fwd_trans_msg = logsumexp(eta_A.T + eta_in[None, :], axis=1)
    fwd_msg = eta_likelihood+fwd_trans_msg
    return fwd_msg, fwd_trans_msg


#@jit
def hmm_bwd_pass(eta_in, eta_current):
    eta_A, eta_likelihood = eta_current
    bwd_trans_msg = logsumexp(eta_A + (eta_likelihood+eta_in)[None, :], axis=1)
    bwd_msg = bwd_trans_msg
    return bwd_msg, bwd_trans_msg


#@jit
def hmm_pw_post(eta_fwd, eta_transition, rho, eta_bwd):
    pw = eta_fwd[:, None]+eta_transition+(rho+eta_bwd)[None, :]
    return pw-logsumexp(pw, keepdims=True)


#@jit
def hmm_inference(params):
    hmm_natparams, eta_prior, eta_transition, qz, qzlag_z = params
    eta_pi, eta_A = hmm_natparams

    # get expected natural parameter messages from lds
    rho = get_rhos((eta_prior, eta_transition), (qz, qzlag_z))

    # broadcast transition natparams over time for ease of use
    eta_A = jnp.repeat(eta_A[None], rho[1:].shape[0], 0)

    # initialize messages
    fwd_msg_init = eta_pi+rho[0]
    bwd_msg_init = jnp.zeros(shape=fwd_msg_init.shape)

    # run message passing
    fwd_trans_msg = tree_prepend(eta_pi, scan(hmm_fwd_pass, fwd_msg_init,
                                              (eta_A, rho[1:]))[1])
    bwd_trans_msg = tree_append(scan(hmm_bwd_pass, bwd_msg_init,
                                     (eta_A, rho[1:]),
                                     reverse=True)[1], bwd_msg_init)
    # compute posterior
    qu = fwd_trans_msg+rho+bwd_trans_msg
    qu = qu-logsumexp(qu, 1, keepdims=True)
    quu = vmap(hmm_pw_post)((fwd_trans_msg+rho)[:-1], eta_A, rho[1:],
                            bwd_trans_msg[1:])
    return (qu, quu)


#@jit
def lds_fwd_pass(in_natparams, current_natparams):
    trans_natparams, likelih_natparams = current_natparams
    eta_in, P_in = in_natparams
    h, J = trans_natparams
    d = eta_in.shape[0]

    # set up gaussian partition
    h_a, h_b = h[:d], h[d:]
    J_aa, J_ab = J[:d, :d], J[:d, d:]
    J_ba, J_bb = J[d:, :d], J[d:, d:]
    # compute fwd pass
    K = invmp(-J_aa-P_in, J_ba.T).T
    fwd_trans_msg = (h_b+K@(h_a+eta_in),
                     J_bb+K@J_ab)
    fwd_msg = tree_sum((fwd_trans_msg, likelih_natparams))
    return fwd_msg, fwd_trans_msg


#@jit
def lds_bwd_pass(in_natparams, current_natparams):
    trans_natparams, likelih_natparams = current_natparams
    h, J = trans_natparams

    # combine incoming messages
    eta_in, P_in = tree_sum((in_natparams, likelih_natparams))

    # set up gaussian partition
    d = eta_in.shape[0]
    h_a, h_b = h[:d], h[d:]
    J_aa, J_ab = J[:d, :d], J[:d, d:]
    J_ba, J_bb = J[d:, :d], J[d:, d:]
    # compute bwd pass
    K = invmp(-J_bb-P_in, J_ab.T).T
    bwd_trans_msg = (h_a+K@(h_b+eta_in),
                     J_aa+K@J_ba)
    return bwd_trans_msg, bwd_trans_msg


#@jit
def pw_posterior(natparams):
    eta_f, P_f, h, J, eta_b, P_b = natparams
    d = eta_f.shape[0]
    eta_pw = jnp.concatenate((eta_f+h[:d], eta_b+h[d:]))
    P_pw = index_add(J, index[:d, :d], P_f)
    P_pw = index_add(P_pw, index[d:, d:], P_b)
    return (eta_pw, P_pw)


#@jit
def lds_inference(z_posteriors, params):
    eta_prior, eta_trans, eta_likelih, qu, n = params
    qu = jnp.exp(qu)
    qz, qzlag_z = z_posteriors

    # computer responsibility weighted natural parameters
    eta_prior_r = get_resp_wgt_natparams(eta_prior, qu[0])
    eta_trans_r = vmap(get_resp_wgt_natparams,
                       in_axes=(None, 0))(eta_trans, qu[1:])

    # compute expected likelihood natparams instead
    E_eta_likelih = get_E_likelihood_natparams(eta_likelih, qz[0], n)

    # initialize messages
    fwd_msg_init = tree_sum((eta_prior_r, tree_get_idx(0, E_eta_likelih)))
    bwd_msg_init = (jnp.zeros(shape=fwd_msg_init[0].shape),
                    jnp.zeros(shape=fwd_msg_init[1].shape))

    # run message passing
    fwd_trans_msgs = tree_prepend(
        eta_prior_r, scan(lds_fwd_pass, fwd_msg_init,
                          (eta_trans_r, tree_dropfirst(E_eta_likelih)))[1]
    )
    bwd_trans_msgs = tree_append(
        scan(lds_bwd_pass, bwd_msg_init,
             (eta_trans_r, tree_dropfirst(E_eta_likelih)), reverse=True)[1],
        bwd_msg_init
    )

    # compute marginal posteriors
    qz_natparams = tree_sum((fwd_trans_msgs, E_eta_likelih,
                             bwd_trans_msgs))
    qzlag_z_natparams = vmap(pw_posterior)(
        jax.tree_leaves(
                   (tree_droplast(tree_sum((fwd_trans_msgs, E_eta_likelih))),
                    eta_trans_r,
                    tree_dropfirst(tree_sum((E_eta_likelih, bwd_trans_msgs))))
        )
    )

    # transform from natparams to mu-precision format
    qz = jax.tree_multimap(lambda a, b: index_update(a, index[n], b),
                           qz, vmap(get_gauss_params)(qz_natparams))
    qzlag_z = jax.tree_multimap(lambda a, b: index_update(a, index[n], b),
                                qzlag_z,
                                vmap(get_gauss_params)(qzlag_z_natparams))
    return (qz, qzlag_z), None


def make_inference(eta_hmm, eta_prior, eta_transition, eta_likelihood):
    #@jit
    def inference(posterior, iteration):
        # unpack
        qz, qzlag_z, qu, quu = posterior

        # run inference
        N = qz[0].shape[0]
        qz, qzlag_z = scan(lds_inference, (qz, qzlag_z),
                           (eta_prior, eta_transition, eta_likelihood,
                           qu, jnp.arange(N)))[0]
        qu, quu = vmap(hmm_inference)(
                       (eta_hmm, eta_prior, eta_transition, qz, qzlag_z))
        return (qz, qzlag_z, qu, quu), None
    return inference


if __name__ == "__main__":
    # generate data
    key = jrandom.PRNGKey(0)
    N = 1
    T = 200
    d = 2
    K = 2
    stay_prob = 0.95
    inference_iters = 20

    # create data
    key, datakey = jrandom.split(key)
    x, z, z_mu, states, *params = gen_slds_linear_ica(N, T, K, d,
                                                      stay_prob, datakey)
    (C, R), (b_prior, Q_prior, B, b, Q), (pi, A) = params

    # initialize posterior distributions
    qkeys = jrandom.split(key, 4)

    qu = jnp.log(jnp.tile(pi[:, None, :], (1, T, 1)))
    quu = jnp.log(jnp.ones((N, T-1, K, K)) / (K**2))
    qz = (jnp.zeros(shape=z.shape),
          jnp.tile(jnp.eye(d)[None, None], (N, T, 1, 1)))
    qzlag_z = (jnp.zeros((N, T-1, 2*d)),
               jnp.tile(jnp.eye(2*d)[None, None], (N, T-1, 1, 1)))

    # run inference
    pdb.set_trace()


    #plt.plot(z[:200])
    #plt.show()
    #plt.plot(z_mu[:200])
    #plt.show()
    #plt.plot(x[:200])
    #plt.show()
