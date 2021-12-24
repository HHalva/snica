import pdb

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as nn
from jax import vmap, jit
from jax.lax import scan, cond
from functools import partial

from utils import multi_tree_stack


def l2normalize(W, axis=0):
    """Normalizes MLP weight matrices.
    Args:
        W (matrix): weight matrix.
        axis (int): axis over which to normalize.
    Returns:
        Matrix l2 normalized over desired axis.
    """
    l2norm = jnp.sqrt(jnp.sum(W*W, axis, keepdims=True))
    W = W / l2norm
    return W


def smooth_leaky_relu(x, alpha=0.1):
    """Calculate smooth leaky ReLU on an input.
    Source: https://stats.stackexchange.com/questions/329776/ \
            approximating-leaky-relu-with-a-differentiable-function
    Args:
        x (float): input value.
        alpha (float): controls level of nonlinearity via slope.
    Returns:
        Value transformed by the smooth leaky ReLU.
    """
    return alpha*x + (1 - alpha)*jnp.logaddexp(x, 0)


def SmoothLeakyRelu(slope):
    """Smooth Leaky ReLU activation function.
    Args:
        slope (float): slope to control degree of non-linearity.
    Returns:
       Lambda function for computing smooth Leaky ReLU.
    """
    return lambda x: smooth_leaky_relu(x, alpha=slope)


def init_layer_params(in_dim, out_dim, key):
    W_key, b_key = jrandom.split(key, 2)
    W_init = nn.initializers.glorot_uniform()
    b_init = nn.initializers.normal()
    return W_init(W_key, (in_dim, out_dim)), b_init(b_key, (out_dim,))


def unif_nica_layer(N, M, key, iter_4_cond=1e4):
    def _gen_matrix(N, M, key):
        A = jrandom.uniform(key, (N, M), minval=0., maxval=2.) - 1.
        A = l2normalize(A)
        cond = jnp.linalg.cond(A)
        return A, cond

    # generate multiple matrices
    keys = jrandom.split(key, iter_4_cond)
    A, conds = vmap(_gen_matrix, (None, None, 0))(N, M, keys)
    target_cond = jnp.percentile(conds, 5)
    target_idx = jnp.argmin(jnp.abs(conds-target_cond))
    return A[target_idx]


def init_nica_params(N, M, nonlin_layers, key):
    '''BEWARE: Assumes factorized distribution
        and equal width in all hidden layers'''
    layer_sizes = [N] + [M]*nonlin_layers + [M]
    keys = jrandom.split(key, len(layer_sizes)-1)
    return [unif_nica_layer(n, m, k) for (n, m, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def init_encoder_params(x_dim, s_dim, hidden_dim, hidden_layers, key):
    '''BEWARE: Assumes factorized distribution
        and equal width in all hidden layers'''
    layer_sizes = [x_dim] + [hidden_dim]*hidden_layers + [s_dim*2]
    keys = jrandom.split(key, len(layer_sizes)-1)
    return [init_layer_params(m, n, k) for (m, n, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def init_decoder_params(x_dim, s_dim, hidden_dim, hidden_layers, key):
    '''BEWARE: Assumes equal width in all hidden layers'''
    layer_sizes = [s_dim] + [hidden_dim]*hidden_layers + [x_dim]
    keys = jrandom.split(key, len(layer_sizes)-1)
    return [init_layer_params(m, n, k) for (m, n, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def make_mlp_fwd_step(activation):
    def mlp_fwd_step(z, layer_params):
        W, b = layer_params
        return activation(z@W + b), None
    return mlp_fwd_step


def make_nica_fwd_step(activation):
    def nica_fwd_step(z, layer_params):
        A = layer_params
        return activation(z@A), None
    return nica_fwd_step


def encoder_mlp(params, x, slope=0.1):
    """Forward pass for encoder MLP that predicts likelihood natparams.
    Args:
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs p(x|s) natparams (v, W) -- see derivation
    """
    activation = SmoothLeakyRelu(slope)
    z = x
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            W, b = hidden_params[i]
            z = activation(z@W + b)
    final_W, final_b = params[-1]
    z = z@final_W + final_b
    v, W_diag = jnp.split(z, 2)
    W = -jnp.diag(nn.softplus(W_diag))
    return v, W


def decoder_mlp(params, x, slope=0.1):
    """Forward pass for encoder MLP for estimating nonlinear mixing function.
    Args:
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs f(s)
    """
    activation = SmoothLeakyRelu(slope)
    z = x
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            W, b = hidden_params[i]
            z = activation(z@W + b)
    final_W, final_b = params[-1]
    z = z@final_W + final_b
    return z


def nica_mlp(params, s, slope=0.1):
    """Forward pass for encoder MLP for estimating nonlinear mixing function.
    Args:
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs f(s)
    """
    assert len(params) > 1
    activation = SmoothLeakyRelu(slope)
    A_in = params[0]
    z = activation(s@A_in)
    if len(params) > 2:
        hidden_params = jnp.stack(params[1:-1])
        fwd_step = make_nica_fwd_step(activation)
        z = scan(fwd_step, z, hidden_params)[0]
    A_final = params[-1]
    z = z@A_final
    return z


if __name__ == "__main__":

    key = jrandom.PRNGKey(0)
    x_dim = 10
    s_dim = 4
    hidden_dim = 20
    hidden_layers = 5
    s = jnp.ones((s_dim,))

    # nonlinear ICA fwd test
    nica_params = init_nica_params(s_dim, x_dim, 3, key)
    x = nica_mlp(nica_params, s, slope=0.1)
    dp = init_decoder_params(x_dim, s_dim, 32, 1, key)
    decoder_mlp(dp, s)
    pdb.set_trace()

    params = init_encoder_params(x_dim, s_dim, hidden_dim,
                                 hidden_layers, key)
    out = encoder_mlp(params, x)

    # linear ICA fwd test
    key = jrandom.PRNGKey(1)
    s_dim = 10
    x_dim = 10
    ica_params = init_nica_params(s_dim, x_dim, 0, key)
    unif_nica_layer(4, 5, key, iter_4_cond=1e3)
