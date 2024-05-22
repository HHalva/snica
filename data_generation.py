import pdb

from jax import config
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

def ar2_lds(alpha, beta, mu, vz, vz0):
    B = jnp.array([[1 + alpha - beta, -alpha], [1.0, .0]])
    b = jnp.array([beta * mu, .0])
    b0 = jnp.array([.0, .0])
    Q = jnp.diag(1 / jnp.array([vz, 1e-3 ** 2]))
    Q0 = jnp.diag(1 / jnp.array([vz0, vz0]))
    return B, b, b0, Q, Q0


def generate_oscillatory_lds(key):
    keys = jrandom.split(key, 3)
    alpha = jrandom.uniform(keys[0], minval=.9, maxval=1.)  # momentum
    beta = .1  # mean reversion
    mu = jrandom.uniform(keys[1], minval=0., maxval=.5)  # mean level
    vz = .003**2  # driving noise variance
    vz0 = .1  # initial state variance
    return ar2_lds(alpha, beta, mu, vz, vz0)


def generate_meanreverting_lds(key):
    alpha = .3  # momentum
    keys = jrandom.split(key, 2)
    beta = jrandom.uniform(keys[0], minval=.5, maxval=.6)  # mean reversion
    mu = jrandom.uniform(keys[1], minval=-.5, maxval=0.)  # mean level
    vz = .3**2  # driving noise variance
    vz0 = .1  # initial state variance
    return ar2_lds(alpha, beta, mu, vz, vz0)


def generate_slds(key):
    os_key, mr_key = jrandom.split(key)
    A = jnp.array([[.99, .01], [.01, .99]])
    a0 = jnp.ones(2)/2
    B, b, b0, Q, Q0 = multi_tree_stack([generate_oscillatory_lds(os_key),
                                        generate_meanreverting_lds(mr_key)])
    return (A, a0), (B, b, b0, Q, Q0)


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
    (A, a0), (B, b, b0, Q, Q0) = generate_slds(p_key)

    # generate hidden markov chain
    states = gen_markov_chain(a0, A, T, s_hmmkey)

    # sample slds
    s_ldskeys = jrandom.split(s_ldskey, T)
    L_diag = jnp.sqrt(vmap(lambda _: 1 / jnp.diag(_))(Q))
    L0_diag = jnp.sqrt(vmap(lambda _: 1 / jnp.diag(_))(Q0))
    z0 = gaussian_sample_w_diag_chol(b0[states[0]], L0_diag[states[0]],
                                     s_ldskeys[0])
    sample_func = make_slds_sampler(B, b, L_diag)
    z, z_mu = tree_prepend((z0, b0[states[0]]),
                           scan(sample_func, z0,
                                (states[1:], s_ldskeys[1:]))[1])
    hmm_params = (a0, A)
    lds_params = (b0, Q0, B, b, Q)
    return z, z_mu, states, lds_params, hmm_params


def gen_slds_nica(N, M, T, L, param_seed, sample_seed, noise_factor=0.1,
                  repeat_layers=False):
    paramkey = jrandom.PRNGKey(param_seed)
    samplekey = jrandom.PRNGKey(sample_seed)
    # generate several slds
    paramkeys = jrandom.split(paramkey, N+1)
    samplekeys = jrandom.split(samplekey, N+1)
    z, z_mu, states, lds_params, hmm_params = vmap(gen_slds, (None, 0, 0), 1)(
        T, paramkeys[1:], samplekeys[1:])
    s = z[:, :, 0]
    # mix signals
    paramkeys = jrandom.split(paramkeys[0], 2)
    nica_params = init_nica_params(N, M, L, paramkeys[0], repeat_layers)
    f = nica_mlp(nica_params, s)
    # add output noise
    noise_var = noise_factor * f.var(0)
    R = jnp.diag(1 / noise_var)
    likelihood_params = (nica_params, R)
    x = f + jnp.sqrt(noise_var)[None,:] * jrandom.normal(samplekeys[0],
                                                             shape=f.shape)
    return x, f, z, z_mu, states, likelihood_params, lds_params, hmm_params


if __name__ == "__main__":
    key = jrandom.PRNGKey(1)
    N = 3
    M = 12
    T = 1000
    L = 2
    pkey, skey = jrandom.split(key)

    # generate data from slds with linear ica
    x, f, z, z_mu, states, x_params, z_params, u_params = gen_slds_nica(
        N, M, T, L, pkey, skey)

    t = 400
    for i in range(N):
        plt.plot(states[:t,i])
        plt.plot(z[:t,i, 0])
        plt.show()

    plt.plot(x[:t])
    plt.show()

    pdb.set_trace()
