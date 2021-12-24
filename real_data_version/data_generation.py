import pdb

from functools import partial

import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit
from jax.lax import scan
from jax.ops import index, index_update
from jax.numpy.linalg import inv

from utils import get_prec_mat, get_mixing_mat, gen_stationary_B
from utils import tree_prepend, invmp, gaussian_sample_from_mu_prec
from utils import multi_tree_stack
from func_estimators import unif_nica_layer, init_nica_params, nica_mlp

import matplotlib.pyplot as plt

from jax.config import config
#config.update("jax_debug_nans", True)


def ar2_lds(alpha, beta, mu, vz, vz0, vx):
    B = jnp.array([[1 + alpha - beta, -alpha], [1.0, .0]])
    b = jnp.array([beta * mu, .0])
    b0 = jnp.array([.0, .0])
    C = jnp.array([[1., .0]])
    c = jnp.array([0.])
    Q = jnp.diag(1 / jnp.array([vz, 1e-3 ** 2]))
    Q0 = jnp.diag(1 / jnp.array([vz0, vz0]))
    R = jnp.diag(1 / jnp.array([vx]))
    return B, b, b0, C, c, Q, Q0, R


def generate_oscillatory_lds(key):
    keys = jrandom.split(key, 3)
    alpha = jrandom.uniform(keys[0], minval=.6, maxval=.8)  # momentum
    beta = .1  # mean reversion
    mu = jrandom.uniform(keys[1], minval=-.5, maxval=.5)  # mean level
    vz = jrandom.uniform(keys[2], minval=.1**2, maxval=.2**2)  # driving noise variance
    vz0 = .1  # initial state variance
    vx = .1**2  # observation noise variance
    return ar2_lds(alpha, beta, mu, vz, vz0, vx)


def generate_meanreverting_lds(key):
    alpha = .1  # momentum
    keys = jrandom.split(key, 2)
    beta = jrandom.uniform(keys[0], minval=.7, maxval=.9)  # mean reversion
    mu = jrandom.uniform(keys[1], minval=-1., maxval=1.)  # mean level
    vz = .2**2  # driving noise variance
    vz0 = .1  # initial state variance
    vx = .1**2  # observation noise variance
    return ar2_lds(alpha, beta, mu, vz, vz0, vx)


def generate_slds(K, key):
    os_key, mr_key = jrandom.split(key)
    A = jnp.array([[.95, .05], [.05, .95]])
    a0 = jnp.ones(K)/K
    B, b, b0, C, c, Q, Q0, R = multi_tree_stack([
        generate_oscillatory_lds(os_key),
        generate_meanreverting_lds(mr_key)])
    return (A, a0), (B, b, b0, C, c, Q, Q0, R)


#@partial(jit, static_argnums=(0, 1))
def gen_markov_chain(pi, A, num_steps, samplekey):

    def make_mc_step(A, num_states):
        @jit
        def sample_mc_step(cur_state, key):
            p = A[cur_state]
            new_state = jrandom.choice(key, jnp.arange(num_states), p=p)
            return new_state, new_state
        return sample_mc_step

    K = pi.shape[0]
    keys = jrandom.split(samplekey, num_steps)
    start_state = jrandom.choice(keys[0], jnp.arange(K), p=pi)
    mc_step = make_mc_step(A, K)
    _, states = scan(mc_step, start_state, keys[1:])
    states = jnp.concatenate([start_state.reshape((1,)), states])
    return states


def make_slds_sampler(B, b, Q):
    @jit
    def sample_slds(z_prev, state_with_key):
        state, key = state_with_key
        z_mu = B[state]@z_prev + b[state]
        z = gaussian_sample_from_mu_prec(z_mu, Q[state], key)
        return z, (z, z_mu)
    return sample_slds


#@partial(jit, static_argnums=(0, 1, 2))
def gen_slds(T, K, d, paramkey, samplekey):
    paramkey, p_key = jrandom.split(paramkey)
    s_ldskey, s_hmmkey = jrandom.split(samplekey)

    # generate variables
    (A, pi), (B, b, b_init, _, _, Q, Q_init, _) = generate_slds(K, p_key)

    # generate hidden markov chain
    states = gen_markov_chain(pi, A, T, s_hmmkey)

    # sample slds
    s_ldskeys = jrandom.split(s_ldskey, T)
    z_init = gaussian_sample_from_mu_prec(b_init[states[0]],
                                          Q_init[states[0]], s_ldskeys[0])
    sample_func = make_slds_sampler(B, b, Q)
    z, z_mu = tree_prepend((z_init, b_init[states[0]]),
                           scan(sample_func, z_init,
                                (states[1:], s_ldskeys[1:]))[1])

    hmm_params = (pi, A)
    lds_params = (b_init, Q_init, B, b, Q)
    return z, z_mu, states, lds_params, hmm_params


def make_linear_mixer(C, R):
    #@jit
    def _mix(carry, s_and_key):
        s, key = s_and_key
        x = gaussian_sample_from_mu_prec(s@C, R, key)
        return carry, x
    return _mix


def make_nonlinear_mixer(nica_params, R):
    #@jit
    def _mix(carry, s_and_key):
        s, key = s_and_key
        y = nica_mlp(nica_params, s)
        x = gaussian_sample_from_mu_prec(y, R, key)
        return carry, x
    return _mix


#@partial(jit, static_argnums=(0, 1, 2, 3))
def gen_slds_linear_ica(N, M, T, K, d, paramkey, samplekey):
    ''' NOTE d has no effect here as assume d=2, also preferably K=2'''
    # generate several slds
    paramkeys = jrandom.split(paramkey, N+1)
    samplekeys = jrandom.split(samplekey, N+1)
    z, z_mu, states, lds_params, hmm_params = vmap(
        gen_slds, (None, None, None, 0, 0))(T, K, d, paramkeys[1:],
                                            samplekeys[1:])
    s = z[:, :, 0]
    # mix signals
    paramkeys = jrandom.split(paramkeys[0], 2)
    C = unif_nica_layer(N, M, paramkeys[0])
    R = jnp.eye(M)*100
    likelihood_params = (C, R)
    signal_mixer = make_linear_mixer(C, R)
    x_keys = jrandom.split(samplekeys[0], T)
    x = scan(signal_mixer, None, (s.T, x_keys))[1].T
    Rxvar_ratio = jnp.mean(jnp.diag(invmp(R, jnp.eye(R.shape[0]))
                                    / jnp.cov(x)))
    # check there is sufficient variance
    print(' *inv(R)/xvar: ', Rxvar_ratio)
    return x, z, z_mu, states, likelihood_params, lds_params, hmm_params


def gen_slds_nica(N, M, T, K, d, L, paramkey, samplekey):
    # generate several slds
    paramkeys = jrandom.split(paramkey, N+1)
    samplekeys = jrandom.split(samplekey, N+1)
    z, z_mu, states, lds_params, hmm_params = vmap(
        gen_slds, (None, None, None, 0, 0))(T, K, d, paramkeys[1:],
                                            samplekeys[1:])
    s = z[:, :, 0]
    # mix signals
    paramkeys = jrandom.split(paramkeys[0], 2)
    nica_params = init_nica_params(N, M, L, paramkeys[0])
    R = jnp.eye(M)*100
    likelihood_params = (nica_params, R)
    signal_mixer = make_nonlinear_mixer(nica_params, R)
    x_keys = jrandom.split(samplekeys[0], T)
    x = scan(signal_mixer, None, (s.T, x_keys))[1].T
    Rxvar_ratio = jnp.mean(jnp.diag(invmp(R, jnp.eye(R.shape[0]))
                                    / jnp.cov(x)))
    # check there is sufficient variance
    print(' *inv(R)/xvar: ', Rxvar_ratio)
    return x, z, z_mu, states, likelihood_params, lds_params, hmm_params


if __name__ == "__main__":
    key = jrandom.PRNGKey(5)
    N = 4
    M = 4
    T = 1000
    d = 2
    K = 2
    L = 3
    #stay_prob = 0.95

    pkey, skey = jrandom.split(key)

    # generate data from slds with linear ica
    x, z, z_mu, states, x_params, z_params, u_params = gen_slds_linear_ica(
        N, M, T, K, d, pkey, skey)

    # with nonlinear ICA
    x, z, z_mu, states, x_params, z_params, u_params = gen_slds_ica(
        N, M, T, K, d, L, pkey, skey)

    # sanity check to make sure Q reasonable level
    b0, Q0, B, b, Q = z_params
    for i in range(N):
        for j in range(K):
            print("* QZvar_ratio:",
                  jnp.diag(inv(Q)[i, j]
                           / jnp.cov(z[i][states[i] == j].T)).mean())

    pdb.set_trace()

    #print(' * inv(Q)/zvar {vr0:.3f} inv(R)/xvar {vr1:.3f}'
    #      .format(vr0=var_ratios[0], vr1=var_ratios[1]))

    #plt.plot(z[:200])
    #plt.show()
    #plt.plot(z_mu[:200])
    #plt.show() #plt.plot(x[:200])
    #plt.show()
