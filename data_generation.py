import pdb

from jax.config import config
config.update("jax_enable_x64", True)


from jax import jit, vmap
from jax.lax import scan
import jax.numpy as jnp
from jax.numpy.linalg import inv
import jax.random as jrandom
import matplotlib.pyplot as plt

from func_estimators import init_nica_params, nica_mlp
from utils import gaussian_sample_w_diag_chol, invmp, tree_prepend
from utils import multi_tree_stack

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
    mu = jrandom.uniform(keys[1], minval=-1., maxval=1.)  # mean level
    vz = jrandom.uniform(keys[2], minval=.1**2, maxval=.2**2)  # driving noise variance
    vz0 = .1  # initial state variance
    vx = .1**2  # observation noise variance
    return ar2_lds(alpha, beta, mu, vz, vz0, vx)


def generate_meanreverting_lds(key):
    alpha = .1  # momentum
    keys = jrandom.split(key, 2)
    beta = jrandom.uniform(keys[0], minval=.7, maxval=.9)  # mean reversion
    mu = jrandom.uniform(keys[1], minval=-3., maxval=3.)  # mean level
    vz = .2**2  # driving noise variance
    vz0 = .1  # initial state variance
    vx = .1**2  # observation noise variance
    return ar2_lds(alpha, beta, mu, vz, vz0, vx)


def generate_slds(key):
    os_key, mr_key = jrandom.split(key)
    A = jnp.array([[.99, .01], [.01, .99]])
    a0 = jnp.ones(2)/2
    B, b, b0, C, c, Q, Q0, R = multi_tree_stack([
        generate_oscillatory_lds(os_key),
        generate_meanreverting_lds(mr_key)])
    return (A, a0), (B, b, b0, C, c, Q, Q0, R)


def gen_markov_chain(a0, A, num_steps, samplekey):
    def make_mc_step(A):
        @jit
        def sample_mc_step(cur_state, key):
            p = A[cur_state]
            new_state = jrandom.choice(key, jnp.arange(2), p=p)
            return new_state, new_state
        return sample_mc_step

    keys = jrandom.split(samplekey, num_steps)
    start_state = jrandom.choice(keys[0], jnp.arange(2), p=a0)
    mc_step = make_mc_step(A)
    _, states = scan(mc_step, start_state, keys[1:])
    states = jnp.concatenate([start_state.reshape((1,)), states])
    return states


def make_slds_sampler(B, b, L_diag):
    @jit
    def sample_slds(z_prev, state_with_key):
        state, key = state_with_key
        z_mu = B[state]@z_prev + b[state]
        z = gaussian_sample_w_diag_chol(z_mu, L_diag[state], key)
        return z, (z, z_mu)
    return sample_slds


def gen_slds(T, paramkey, samplekey):
    paramkey, p_key = jrandom.split(paramkey)
    s_ldskey, s_hmmkey = jrandom.split(samplekey)

    # generate variables
    (A, a0), (B, b, b0, _, _, Q, Q_init, _) = generate_slds(p_key)

    # generate hidden markov chain
    states = gen_markov_chain(a0, A, T, s_hmmkey)

    # sample slds
    s_ldskeys = jrandom.split(s_ldskey, T)
    L_diag = jnp.sqrt(vmap(lambda _: 1 / jnp.diag(_))(Q))
    L0_diag = jnp.sqrt(vmap(lambda _: 1 / jnp.diag(_))(Q0))
    z_init = gaussian_sample_w_diag_chol(b0[states[0]], L0_diag,
                                         s_ldskeys[0])
    sample_func = make_slds_sampler(B, b, L_diag)
    z, z_mu = tree_prepend((z_init, b0[states[0]]),
                           scan(sample_func, z_init,
                                (states[1:], s_ldskeys[1:]))[1])
    hmm_params = (a0, A)
    lds_params = (b0, Q_init, B, b, Q)
    return z, z_mu, states, lds_params, hmm_params


def gen_slds_nica(N, M, T, L, paramkey, samplekey, repeat_layers=False):
    # generate several slds
    paramkeys = jrandom.split(paramkey, N+1)
    samplekeys = jrandom.split(samplekey, N+1)
    z, z_mu, states, lds_params, hmm_params = vmap(
        gen_slds, (None, None, None, 0, 0))(T, K, d, paramkeys[1:],
                                            samplekeys[1:])
    s = z[:, :, 0]
    # mix signals
    paramkeys = jrandom.split(paramkeys[0], 2)
    nica_params = init_nica_params(N, M, L, paramkeys[0], repeat_layers)
    f = vmap(nica_mlp, (None, 1), 1)(nica_params, s)
    # add appropriately scaled output noise (R is precision!)
    R = inv(jnp.eye(M)*jnp.diag(jnp.cov(f))*0.15)
    likelihood_params = (nica_params, R)
    x_keys = jrandom.split(samplekeys[0], T)
    x = vmap(gaussian_sample_from_mu_prec, (1, None, 0), 1)(f, R, x_keys)
    # double-check variance levels on output noise
    Rxvar_ratio = jnp.mean(jnp.diag(invmp(R, jnp.eye(R.shape[0]))
                                    / jnp.cov(x)))
    print(' *inv(R)/xvar: ', Rxvar_ratio)
    return x, f, z, z_mu, states, likelihood_params, lds_params, hmm_params


if __name__ == "__main__":
    for i in range(10):
        key = jrandom.PRNGKey(i)
        N = 3
        M = 12
        T = 100000
        d = 2
        K = 2
        L = 1
        #stay_prob = 0.95

        pkey, skey = jrandom.split(key)

        # generate data from slds with linear ica
        if L == 0:
            x, z, z_mu, states, x_params, z_params, u_params = gen_slds_linear_ica(
                N, M, T, K, d, pkey, skey)
        elif L > 0:
        # with nonlinear ICA
            x, z, z_mu, states, x_params, z_params, u_params = gen_slds_nica(
                N, M, T, K, d, L, pkey, skey)
