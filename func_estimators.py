from jax.config import config

config.update("jax_enable_x64", True)

import pdb

import jax.nn as nn
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
from jax.lax import scan
from jax.ops import index, index_update


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


def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x


def init_layer_params(in_dim, out_dim, key):
    W_key, b_key = jrandom.split(key, 2)
    W_init = nn.initializers.glorot_uniform(dtype=jnp.float64)
    b_init = nn.initializers.normal(dtype=jnp.float64)
    return W_init(W_key, (in_dim, out_dim)), b_init(b_key, (out_dim,))


def unif_nica_layer(N, M, key, iter_4_cond=1e4):
    def _gen_matrix(N, M, key):
        A = jrandom.uniform(key, (N, M), minval=0., maxval=2.) - 1.
        A = l2normalize(A)
        _cond = jnp.linalg.cond(A)
        return A, _cond

    # generate multiple matrices
    keys = jrandom.split(key, iter_4_cond)
    A, conds = vmap(_gen_matrix, (None, None, 0))(N, M, keys)
    target_cond = jnp.percentile(conds, 25)
    target_idx = jnp.argmin(jnp.abs(conds-target_cond))
    return A[target_idx]


def init_nica_params(N, M, nonlin_layers, key, repeat_layers):
    '''BEWARE: Assumes factorized distribution
        and equal width in all hidden layers'''
    layer_sizes = [N] + [M]*nonlin_layers + [M]
    keys = jrandom.split(key, len(layer_sizes)-1)
    if repeat_layers:
        _keys = keys
        keys = jnp.repeat(_keys[0][None], _keys.shape[0], 0)
        if N != M:
            keys = index_update(keys, index[1:], _keys[-1])
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


def encoder_mlp(params, x, activation='xtanh', slope=0.1):
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
    if activation == 'xtanh':
        act = xtanh(slope)
    else:
        act = SmoothLeakyRelu(slope)
        #act = lambda x: nn.leaky_relu(x, slope)
    z = x
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            W, b = hidden_params[i]
            z = act(z@W + b)
    final_W, final_b = params[-1]
    z = z@final_W + final_b
    v, W_diag = jnp.split(z, 2)
    W = -jnp.diag(nn.softplus(W_diag))
    return v, W


def decoder_mlp(params, x, activation='xtanh', slope=0.1):
    """Forward pass for encoder MLP for estimating nonlinear mixing function.
    Args: (IGNORE; OLD)
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs f(s)
    """
    if activation == 'xtanh':
        act = xtanh(slope)
    else:
        act = SmoothLeakyRelu(slope)
        #act = lambda x: nn.leaky_relu(x, slope)
    z = x
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            W, b = hidden_params[i]
            z = act(z@W + b)
    final_W, final_b = params[-1]
    z = z@final_W + final_b
    return z


def nica_mlp(params, s, activation='xtanh', slope=0.1):
    """Forward pass for encoder MLP for estimating nonlinear mixing function.
    Args: (OLD; IGNORE)
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs f(s)
    """
    if activation == 'xtanh':
        act = xtanh(slope)
    else:
        act = SmoothLeakyRelu(slope)
    z = s
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            z = act(z@hidden_params[i])
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
    nica_params = init_nica_params(s_dim, x_dim, 3, key, repeat_layers=False)
    x = nica_mlp(nica_params, s, slope=0.1)
    dp = init_decoder_params(x_dim, s_dim, 32, 1, key)
    decoder_mlp(dp, s)

    params = init_encoder_params(x_dim, s_dim, hidden_dim,
                                 hidden_layers, key)
    out = encoder_mlp(params, x)

    pdb.set_trace()

    # linear ICA fwd test
    key = jrandom.PRNGKey(1)
    s_dim = 10
    x_dim = 10
    ica_params = init_nica_params(s_dim, x_dim, 0, key, repeat_layers=False)
    unif_nica_layer(4, 5, key, iter_4_cond=1e3)
