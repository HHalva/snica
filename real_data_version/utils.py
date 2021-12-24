import pdb

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from jax import random as jrandom
from jax import vmap
from jax import jit, lax
from jax.ops import index_update, index_add, index
from jax.lax import scan
from jax.numpy.linalg import inv

import scipy as sp
import matplotlib.pyplot as plt

#@partial(jit, static_argnums=(0,))
def get_prec_mat(n, prec_scale, key):
    '''Create a random covariance matrix
        prec_scale: scale precisions by a factor
        key: JAX PRNG key
    '''
    def _cond(prec_with_key):
        prec, _ = prec_with_key
        return jnp.linalg.det(prec) < 0

    def _make_prec_sampler(n):
        def sample_prec(prec_with_key):
            _, key = prec_with_key
            key, _key = jrandom.split(key)
            mat = jrandom.normal(_key, shape=(n, n))
            prec_mat = (0.5*(mat + mat.T) + jnp.eye(n)*n)*prec_scale
            return (prec_mat, key)
        return sample_prec

    mat = jrandom.normal(key, shape=(n, n))
    prec_mat = (0.5*(mat + mat.T) + jnp.eye(n)*n)*prec_scale
    prec_sampler = _make_prec_sampler(n)
    prec_mat, _ = lax.while_loop(_cond, prec_sampler, (prec_mat, key))
    return prec_mat


@partial(jit, static_argnums=(0,))
def get_mixing_mat(n, key, iters=1000):
    '''Create a linear mixing matrices with their condition number
        n: dimension of the (square) matrix
        key: JAX PRNG key
    '''
    def _make_sampler(n):
        def __sample_with_cond_num(carry, key):
            mat = jrandom.uniform(key, shape=(n, n),
                                  minval=-1, maxval=1)
            mat = mat / mat.sum(1, keepdims=True)
            cond = jnp.linalg.cond(mat)
            return carry, (cond, mat)
        return __sample_with_cond_num

    keys = jrandom.split(key, iters)
    sampler = _make_sampler(n)
    conds, mats = scan(sampler, None, keys)[1]
    return conds, mats


def gen_stationary_B(Q, key):
    '''Given a precision matrix Q, create a stationary B, and compute
    the ratio of variance Q explains in the resulting process'''
    d = Q.shape[0]
    B = jrandom.uniform(key, shape=(d, d),
                        minval=-1., maxval=3.)
    B = B / (1.1*jnp.abs(B).sum(1, keepdims=True))
    Q_tot_diag_var = jnp.median(jnp.abs(jnp.diag(inv(Q))))
    obs_var = inv(jnp.eye(d**2)-jnp.kron(B, B)) @ inv(Q).flatten()
    obs_var = obs_var.reshape((d, d))
    obs_tot_diag_var = jnp.median(jnp.abs(jnp.diag(obs_var)))
    Q_obs_var_ratio = Q_tot_diag_var / obs_tot_diag_var
    return Q, (Q_obs_var_ratio, B)


def init_params(N, T, d, K, key):
    # initialize hmm parameters
    pi = jnp.ones((N, K))/K
    A = jnp.ones((N, K, K))/K

    # initialize LDS parameters
    key, *subkeys = jrandom.split(key, 6)
    B = jrandom.uniform(subkeys[0], shape=(N, K, d, d),
                        minval=-1., maxval=3.)
    B = B / (1.1*jnp.abs(B).sum(-1, keepdims=True))
    b = jrandom.uniform(subkeys[1], shape=(N, K, d), minval=-10, maxval=10)
    Q = jnp.tile(get_prec_mat(d, prec_scale=0.001, key=subkeys[2]), (N, K, 1, 1))
    b_init = jrandom.uniform(subkeys[3], shape=(N, K, d),
                             minval=-10, maxval=10)
    Q_init = jnp.tile(get_prec_mat(d, prec_scale=0.001, key=subkeys[4]),
                      (N, K, 1, 1))

    # initialize mixing parameters
    key, *subkeys = jrandom.split(key, 3)
    cond_nums, mix_mats = get_mixing_mat(N, subkeys[0])
    C = mix_mats[jnp.argmin(cond_nums)]
    R = get_prec_mat(N, prec_scale=0.001, key=subkeys[1])
    return C, R, b_init, Q_init, B, b, Q, pi, A


def get_hmm_natparams(carry, hmm_params):
    pi, A = hmm_params
    eta_pi = jnp.log(pi)
    eta_A = jnp.log(A)
    return carry, (eta_pi, eta_A)


def get_prior_natparams(carry, params):
    b_init_k, Q_init_k = params
    h_init_k = Q_init_k @ b_init_k
    J_init_k = -0.5*Q_init_k
    return carry, (h_init_k, J_init_k)


def get_transition_natparams(carry, params):
    B_k, b_k, Q_k = params
    d = B_k.shape[0]
    Jaa = B_k.T@Q_k@B_k
    Jab = -B_k.T@Q_k
    Jba = -Q_k@B_k
    Jbb = Q_k
    J_k = -0.5*jnp.vstack((jnp.hstack((Jaa, Jab)), jnp.hstack((Jba, Jbb))))
    h_k = (-2*J_k)@jnp.concatenate((jnp.zeros((d,)), b_k))
    return carry, (h_k, J_k)


def get_likelihood_natparams(C, R, x):
    v = C.T@R@x
    W = -0.5*C.T@R@C
    # repeate for ease of use
    W = jnp.repeat(W[:, :, None], v.shape[1], -1)
    return (v, W)


def get_E_prior_natparams(qz_prior, eta_prior):
    h_prior, J_prior = eta_prior
    Ez, Ez_outer = get_expected_suffstats(None, qz_prior)[1]
    lognorm = 0.25*jnp.trace(invmp(-J_prior, jnp.outer(h_prior, h_prior)))
    lognorm = lognorm+0.5*jnp.linalg.slogdet(
        jnp.pi*invmp(-J_prior, jnp.eye(J_prior.shape[0])))[1]
    rho_prior = h_prior@Ez+jnp.trace(J_prior@Ez_outer)-lognorm
    return rho_prior


def get_E_transition_natparams(qzlag_z, eta_transition):
    h, J = eta_transition
    h_a, h_b = jnp.split(h, 2)
    d = h_a.shape[0]
    Ezlag_z, Ezlag_z_outer = get_expected_suffstats(None, qzlag_z)[1]
    _Q = -2*J[d:, d:]
    _V = invmp(_Q, jnp.eye(_Q.shape[0]))
    _b = _V@h_b
    lognorm = 0.5*(_b.T@_Q@_b+jnp.linalg.slogdet(2*jnp.pi*_V)[1])
    rho = h@Ezlag_z+jnp.trace(J@Ezlag_z_outer)-lognorm
    return rho


def get_rhos(lds_natparams, lds_posteriors):
    eta_prior, eta_transition = lds_natparams
    qz, qzlag_z = lds_posteriors
    rho_prior = vmap(get_E_prior_natparams, (None, 0))(tree_get_idx(0, qz),
                                                       eta_prior)
    rho_transition = vmap(vmap(get_E_transition_natparams, (None, 0)),
                          (0, None))(qzlag_z, eta_transition)
    return jnp.vstack((rho_prior, rho_transition))


def get_E_likelihood_natparams(likelih_natparams, qz_mu, n):
    d = qz_mu.shape[-1]
    v_n, W_n = likelih_natparams
    qs_mu = qz_mu[:, :, 0]
    E_v_n = index_update(jnp.zeros((v_n.shape[0], d)), index[:, 0], v_n)
    E_v_n = index_add(E_v_n, index[:, 0],
                      2*((W_n*qs_mu).sum(0)-W_n[n]*qs_mu[n]))
    E_W_n = index_update(jnp.zeros((v_n.shape[0], d, d)),
                         index[:, 0, 0], W_n[n])
    return E_v_n, E_W_n


def get_resp_weighed_natparams(natparams, qu):
    h, J = natparams
    h_resp = jnp.sum(qu[:, None]*h, 0)
    J_resp = jnp.sum(qu[:, None, None]*J, 0)
    return natparams, (h_resp, J_resp)


def get_gauss_params(carry, natparams):
    eta, P = natparams
    mu = 0.5*invmp(-P, jnp.eye(P.shape[0]))@eta
    prec = -2*P
    return carry, (mu, prec)


def get_expected_suffstats(carry, qz):
    Ez = qz[0]
    Ez_outer = invmp(qz[1], jnp.eye(qz[1].shape[0])) + jnp.outer(Ez, Ez)
    return carry, (Ez, Ez_outer)


def tree_expected_suffstats(carry, posteriors):
    tree_E_suffstats = jax.tree_map(lambda a:
                                    get_expected_suffstats(None, a)[1],
                                    posteriors)
    return carry, tree_E_suffstats


def vrepeat_tuple(tpl, T):
    return scan(lambda tup, _: (tup, tup), tpl, None, length=T)[1]


def tree_prepend(prep, tree):
    preprended = jax.tree_multimap(
        lambda a, b: jnp.vstack((a[None], b)), prep, tree
    )
    return preprended


def tree_append(tree, app):
    appended = jax.tree_multimap(
        lambda a, b: jnp.vstack((a, b[None])), tree, app
    )
    return appended


def tree_sum(trees):
    '''Sum over pytrees'''
    return jax.tree_multimap(lambda *x: sum(x), *trees)

def tree_sub(tree1, tree2):
    return jax.tree_multimap(
        lambda a, b: a-b, tree1, tree2)

def tree_droplast(tree):
    '''Drop last index from each leaf'''
    return jax.tree_map(lambda a: a[:-1], tree)


def tree_dropfirst(tree):
    '''Drop first index from each leaf'''
    return jax.tree_map(lambda a: a[1:], tree)


def tree_get_idx(idx, tree):
    '''Get idx row from each leaf of tuple'''
    return jax.tree_map(lambda a: a[idx], tree)


def multi_tree_stack(trees):
    '''Stack trees along a new axis'''
    return jax.tree_multimap(lambda *a: jnp.stack(a), *trees)


# inv(L*L.T)*Y
def invcholp(L, Y):
    D = jax.scipy.linalg.solve_triangular(L, Y, lower=True)
    B = jax.scipy.linalg.solve_triangular(L.T, D, lower=False)
    return B


# inv(X)*Y
def invmp(X, Y):
    return invcholp(jnp.linalg.cholesky(X), Y)


def gaussian_sample_from_mu_prec(mu, prec, key):
    L = jnp.linalg.cholesky(prec)
    z = jrandom.normal(key, mu.shape)
    return mu+jax.scipy.linalg.solve_triangular(L, z, lower=True)


def matching_sources_corr(est_sources, true_sources, method="pearson"):
    """Finding matching indices between true and estimated sources.
    Args:
        est_sources (array): data on estimated independent components.
        true_sources (array): data on true independent components.
        method (str): "pearson" or "spearman" correlation method to use.
    Returns:
        mean_abs_corr (array): average correlation matrix between
                               matched sources.
        s_est_sort (array): estimed sources array but columns sorted
                            according to best matching index.
        cid (array): vector of the best matching indices.
    """
    N = est_sources.shape[0]

    # calculate correlations
    if method == "pearson":
        corr = np.corrcoef(true_sources, est_sources, rowvar=True)
        corr = corr[0:N, N:]
    elif method == "spearman":
        corr, pvals = sp.stats.spearmanr(true_sources, est_sources, axis=1)
        corr = corr[0:N, N:]

    # sort variables to try find matching components
    ridx, cidx = sp.optimize.linear_sum_assignment(-np.abs(corr))

    # calc with best matching components
    mean_abs_corr = np.mean(np.abs(corr[ridx, cidx]))
    s_est_sorted = est_sources[cidx, :]
    return mean_abs_corr, s_est_sorted, cidx


def plot_ic(u, z_mu, qu, qz_mu, qz_prec, ax0, ax1):
    T, K = qu.shape
    qz_var = vmap(lambda a: invmp(a, jnp.eye(a.shape[0])))(qz_prec)
    qz_sd = jnp.sqrt(qz_var[:, 0, 0])

    ax0.clear()
    ax1.clear()
    ax0.imshow(qu.T, aspect='auto', interpolation='none')
    ax0.set_xlim([0, T])
    ax0.axis('off')

    switches = jnp.concatenate([jnp.array([0]),
                                jnp.arange(1, T)[u[:-1] != u[1:]],
                                jnp.array([T])])
    # expand the colour map if K > 4
    cmap = jnp.array([[1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0]])
    for i in range(len(switches)-1):
        ax1.axvspan(switches[i], switches[i+1], alpha=0.25,
                    color=cmap[u[switches[i]]])
    ax1.plot(z_mu[:, 0], color='blue')
    ax1.plot(qz_mu[:, 0], color='red')
    ax1.fill_between(jnp.arange(0, T),
                     qz_mu[:, 0] - qz_sd, qz_mu[:, 0] + qz_sd, alpha=0.5)
    ax1.set_xlim([0, T])


def nsym_grad(cov_g):
    '''Nonstandard symmetrization operator'''
    return cov_g+cov_g.T-jnp.eye(cov_g.shape[0])*cov_g


def sym_grad(cov_g):
    '''standard symmetrization operator'''
    return 0.5*(cov_g+cov_g.T)


if __name__ == "__main__":
    # create covariance matrix
    key = jrandom.PRNGKey(0)
    prec = jnp.zeros((5, 5))

    # test sampling on sphere
    keys = jrandom.split(jrandom.PRNGKey(1), 1000)
    sample = vmap(lambda a, b, c: gaussian_sample_from_mu_prec(a, b, c),
                  (None, None, 0))(jnp.ones((2,)), jnp.eye(2)/10, keys)
    pdb.set_trace()

