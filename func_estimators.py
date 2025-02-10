import pdb

import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, lax, jit

from utils import jax_print

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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


def smooth_leaky_relu(slope):
    """Calculate smooth leaky ReLU on an input.
    Source: https://stats.stackexchange.com/questions/329776/ \
            approximating-leaky-relu-with-a-differentiable-function
    Args:
        x (float): input value.
        alpha (float): controls level of nonlinearity via slope.
    Returns:
        Value transformed by the smooth leaky ReLU.
    """
    return lambda x: slope*x + (1 - slope)*jnp.logaddexp(x, 0)


def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x


def init_layer_params(in_dim, out_dim, key):
    W_key, b_key = jrandom.split(key, 2)
    W_init = nn.initializers.glorot_uniform(dtype=jnp.float64)
    b_init = nn.initializers.normal(dtype=jnp.float64)
    return W_init(W_key, (in_dim, out_dim)), b_init(b_key, (out_dim,))


def unif_nica_layer(N, M, key, iter_4_cond=1e4):
    def _gen_matrix(N, M, key):
        A = jrandom.uniform(key, (N, M), minval=-1., maxval=1.)
        A = l2normalize(A)
        _cond = jnp.linalg.cond(A)
        return A, _cond

    # generate multiple matrices
    keys = jrandom.split(key, int(iter_4_cond))
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
        _keys = keys.copy()
        keys = jnp.repeat(_keys[0][None], _keys.shape[0], 0)
        if N != M:
            keys = keys.at[1:].set(_keys[-1])
    return [unif_nica_layer(n, m, k) for (n, m, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


#def init_encoder_params(x_dim, s_dim, hidden_dim, hidden_layers, key):
#    '''BEWARE: Assumes factorized distribution
#        and equal width in all hidden layers'''
#    layer_sizes = [x_dim] + [hidden_dim]*hidden_layers + [s_dim*2]
#    keys = jrandom.split(key, len(layer_sizes)-1)
#    return [init_layer_params(m, n, k) for (m, n, k)
#            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
#
#
#def init_decoder_params(x_dim, s_dim, hidden_dim, hidden_layers, key):
#    '''BEWARE: Assumes equal width in all hidden layers'''
#    layer_sizes = [s_dim] + [hidden_dim]*hidden_layers + [x_dim]
#    keys = jrandom.split(key, len(layer_sizes)-1)
#    return [init_layer_params(m, n, k) for (m, n, k)
#            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
#
#
#def encoder_mlp(params, x, activation='xtanh', slope=0.1):
#    """Forward pass for encoder MLP that predicts likelihood natparams.
#    Args:
#        params (list): nested list where each element is a list of weight
#            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
#        inputs (matrix): input data.
#        slope (float): slope to control the nonlinearity of the activation
#            function.
#    Returns:
#        Outputs p(x|s) natparams (v, W) -- see derivation
#    """
#    if activation == 'xtanh':
#        act = xtanh(slope)
#    else:
#        act = SmoothLeakyRelu(slope)
#        #act = lambda x: nn.leaky_relu(x, slope)
#    z = x
#    if len(params) > 1:
#        hidden_params = params[:-1]
#        for i in range(len(hidden_params)):
#            W, b = hidden_params[i]
#            z = act(z@W + b)
#    final_W, final_b = params[-1]
#    z = z@final_W + final_b
#    v, W_diag = jnp.split(z, 2)
#    W = -jnp.diag(nn.softplus(W_diag))
#    return v, W
#
#
#def decoder_mlp(params, x, activation='xtanh', slope=0.1):
#    """Forward pass for encoder MLP for estimating nonlinear mixing function.
#    Args: (IGNORE; OLD)
#        params (list): nested list where each element is a list of weight
#            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
#        inputs (matrix): input data.
#        slope (float): slope to control the nonlinearity of the activation
#            function.
#    Returns:
#        Outputs f(s)
#    """
#    if activation == 'xtanh':
#        act = xtanh(slope)
#    else:
#        act = smooth_leaky_relu(slope)
#        #act = lambda x: nn.leaky_relu(x, slope)
#    z = x
#    if len(params) > 1:
#        hidden_params = params[:-1]
#        for i in range(len(hidden_params)):
#            W, b = hidden_params[i]
#            z = act(z@W + b)
#    final_W, final_b = params[-1]
#    z = z@final_W + final_b
#    return z


@jit
def nica_mlp(params, s, xtanh_act=True, slope=0.01):
    """Forward-pass of mixing function.
    """

    def _fwd_pass(z, params_list):
        for A in params_list[:-1]:
            in_dim, out_dim = A.shape
            # layer norm
            #_z = layer_norm(z)
            # nonlinear activation
            _z = lax.cond(xtanh_act, xtanh(slope),
                           smooth_leaky_relu(slope), z @ A)
            # residual connection
            z = jnp.eye(out_dim, in_dim)@z + _z
        return z + z@params_list[-1]


    z = lax.cond(len(params) > 1, lambda z, B: _fwd_pass(z, B),
                 lambda z, B: z @ B[0], s, params)
    return z


def layer_norm(x, beta=0, gamma=1, eps=1e-5):
    mu = x.mean()
    sigma = x.std()
    return (x - mu)*gamma / (sigma+eps) + beta


def color_wheel(s):
    # Convert from JAX arrays to NumPy for color computations
    s_np = np.array(s)

    # Separate coordinates
    x_vals = s_np[:, 0]
    y_vals = s_np[:, 1]

    # 1) Compute angle theta in [-pi, pi]
    theta = np.arctan2(y_vals, x_vals)

    # Let's cycle through the color wheel multiple times for more distinct hues
    color_cycles = 1  # Increase this number for even more color repeats
    # (-pi => hue=0, +pi => hue=1) multiplied by 'color_cycles'
    hue = (theta + np.pi) / (2 * np.pi) * color_cycles
    hue = np.mod(hue, 1.0)  # Wrap back into [0, 1] range

    # 2) Compute radius for brightness
    r = np.sqrt(x_vals**2 + y_vals**2)
    r_max = r.max()
    value = np.clip(r / (r_max + 1e-7), 0, 1)  # Avoid division by zero

    # Keep saturation=1 for vivid colors
    saturation = np.ones_like(hue)

    # Stack HSV and convert to RGB
    hsv = np.stack([hue, saturation, value], axis=1)
    rgb_colors=mcolors.hsv_to_rgb(hsv)

    # plot
    plt.scatter(x_vals, y_vals, c=rgb_colors, s=20, alpha=0.8)

    plt.axvline(0, color='gray', linewidth=0.5)
    plt.gca().set_aspect('equal', 'box')  # Make x & y scales the same
    plt.title('2D Normal Points with Angle- and Radius-Based Coloring')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    x_dim = 2
    s_dim = 2
    key = jrandom.PRNGKey(0)
    key2, _ = jrandom.split(key)
    s = jrandom.normal(key, shape=(100000, s_dim))
    hidden_layers = 10

    # nonlinear ICA fwd test
    nica_params = init_nica_params(s_dim, x_dim, hidden_layers, key2,
                                   repeat_layers=False)
    x = vmap(lambda _: nica_mlp(nica_params, _, slope=0.01))(s)


    color_wheel(s)

    #plt.scatter(s.T[0], s.T[1])
    #plt.show()
    #plt.scatter(x.T[0], x.T[1])
    #plt.show()



