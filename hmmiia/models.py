import pdb
import jax.numpy as jnp
import numpy as np

from jax import random as jrandom
from jax import nn as jnn

from utils import l2normalize, find_mat_cond_thresh, SmoothLeakyRelu
from subfunc.showdata import *


def unif_invertible_layer_weights(key, in_dim, out_dim, lrelu_slope, in_double, w_cond_thresh,
                                  weight_range, bias_range):
    """Create square random weight matrix and bias with uniform
    initialization and good condition number.

    Args:
        key: JAX random key.
        in_dim (int): layer input dimension.
        out_dim (int): layer output dimension.
        w_cond_thresh (float/None): upper threshold for condition number for
            generated weight matrices. If 'None', then no threshold applied.
        weight_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of weight matrix.
        bias_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of bias vector.

    Returns:
        Tuple (weight matrix, bias vector).
    """
    # ensure good condition number for weight matrix
    W_key, b_key = jrandom.split(key)
    if w_cond_thresh is None:
        W = jrandom.uniform(W_key, (in_dim, out_dim), minval=weight_range[0],
                            maxval=weight_range[1])
        # W = l2normalize(W, 1)
        W = W * np.sqrt(6 / ((1 + lrelu_slope ** 2) * W.shape[0]))
    else:
        cond_W = w_cond_thresh + 1
        while cond_W > w_cond_thresh:
            W_key, subkey = jrandom.split(W_key)
            W = jrandom.uniform(subkey, (in_dim, out_dim),
                                minval=weight_range[0],
                                maxval=weight_range[1])
            # W = l2normalize(W, 1)
            W = W * jnp.sqrt(6 / ((1 + lrelu_slope ** 2) * W.shape[0]))
            cond_W = np.linalg.cond(W)
    # b = jrandom.uniform(b_key, (out_dim,), minval=bias_range[0],
    #                     maxval=bias_range[1])
    b = jnp.zeros(out_dim)

    # concatenate additional W and b for doubled input
    if in_double:
        cond_W = w_cond_thresh + 1
        while cond_W > w_cond_thresh:
            W_key, subkey = jrandom.split(W_key)
            W2 = jrandom.uniform(subkey, (in_dim, out_dim),
                                minval=weight_range[0],
                                maxval=weight_range[1])
            # W2 = l2normalize(W2, 1)
            W2 = W2 * jnp.sqrt(6 / ((1 + lrelu_slope ** 2) * W.shape[0]))
            cond_W = np.linalg.cond(W2)
        # b2 = np.zeros((dim,))
        #
        W = jnp.concatenate([W, W2], axis=0) / np.sqrt(2)
        # b = np.concatenate([b, b2], axis=0)

    return W, b


def init_invertible_mlp_params(key, dim, num_layers,
                               weight_range=[-1., 1.], bias_range=[0., 1.]):
    """Initialize weights and biases of an invertible MLP.

    Note that all weight matrices have equal dimensionalities.

    Args:
        key: JAX random key.
        dim (int): dimensionality of weight matrices.
        num_layers (int): number of layers.
        weight_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of weight matrix.
        bias_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of bias vector.

    Returns:
        Nested list where each element is a list [W, b] that contains
        weight matrix and bias for a given layer.
    """
    keys = jrandom.split(key, num_layers)
    ct = find_mat_cond_thresh(dim, weight_range)
    return [unif_invertible_layer_weights(k, d_in, d_out, slp, in_d, ct, weight_range, bias_range)
            for k, d_in, d_out, slp, in_d in zip(keys, [dim]*num_layers, [dim]*num_layers, [0.1]*(num_layers-1)+[1], [True]+[False]*(num_layers-1))]


def invertible_mlp_fwd(params, x, slope=0.1):
    z = np.zeros_like(x)
    N = z.shape[0]
    for i in range(1, N):
        zi = np.concatenate([z[i-1,:], x[i-1,:]]).reshape([1,-1])
        for W, b in params[:-1]:
            zi = np.dot(zi, np.array(W)) + np.array(b)
            zi[zi < 0] = slope * zi[zi < 0]
        final_W, final_b = params[-1]
        zi = np.dot(zi, final_W) + final_b
        z[i, :] = zi
        # ops.index_update(z, ops.index[i, :], zi.reshape([-1]))
    z = jnp.array(z)
    return z


def invertible_mlp_inverse(params, x, lrelu_slope=0.1):
    """Inverse of invertible MLP defined above.

    Args:
        params (list): list where each element is a list of layer weight
            and bias [W, b]. len(params) is the number of layers.
        x (vector): output of forward MLP, here observed data.
        slope (float): slope for activation function.

    Returns:
        Inputs into the MLP. Here the independent components.
    """
    z = x
    params_rev = params[::-1]
    final_W, final_b = params_rev[0]
    z = z - final_b
    z = jnp.dot(z, jnp.linalg.inv(final_W))
    for W, b in params_rev[1:]:
        z = jnn.leaky_relu(z, 1./lrelu_slope)
        z = z - b
        z = jnp.dot(z, jnp.linalg.inv(W))
    return z


def init_mlp_params(key, layer_sizes):
    """Initialize weight and bias parameters of an MLP.

    Args:
        key: JAX random key.
        sizes (list): list of dimensions for each layer. For example MLP with
            one 10-unit hidden layer and 3-dimensional input and output would
            be [3, 10, 3].

    Returns:
        Nested list where each element is a list of weight matrix and bias for
            that layer [W, b].
    """
    keys = jrandom.split(key, len(layer_sizes))
    in_sizes = layer_sizes[:-1]
    out_sizes = layer_sizes[1:]
    return [unif_invertible_layer_weights(k, m, n, slp, in_d, None,
                                          [-1., 1.], [0., 0.1])
            for k, m, n, slp, in_d in zip(keys, in_sizes, out_sizes,
                                          [1]*len(layer_sizes),
                                          [False]*len(layer_sizes))]


def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x


def mlp(params, inputs, slope=0.1):
    """Forward pass through an MLP with SmoothLeakyRelu activations.

    Args:
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.

    Returns:
        Output of the MLP.
    """
    #activation = SmoothLeakyRelu(slope)
    activation = xtanh(slope)
    z = inputs
    for W, b in params[:-1]:
        z = jnp.matmul(z, W)+b
        z = activation(z)
    final_W, final_b = params[-1]
    z = jnp.matmul(z, final_W) + final_b
    # concatenate xtm1 (IIA)
    if jnp.ndim(inputs) == 1:
        xtm1 = inputs[int(inputs.shape[0]/2):]
    else:
        xtm1 = inputs[:, int(inputs.shape[1]/2):]
    z = jnp.concatenate([z, xtm1], axis=-1)
    return z


# def init_mlp_params(key, layer_sizes):
#     """Initialize weight and bias parameters of an MLP.
#
#     Args:
#         key: JAX random key.
#         sizes (list): list of dimensions for each layer. For example MLP with
#             one 10-unit hidden layer and 3-dimensional input and output would
#             be [3, 10, 3].
#
#     Returns:
#         Nested list where each element is a list of weight matrix and bias for
#             that layer [W, b].
#     """
#     keys = jrandom.split(key, len(layer_sizes))
#     in_sizes = layer_sizes[:-1]
#     out_sizes = layer_sizes[1:]
#     out_sizes[:-1] = list(2 * np.array(out_sizes[:-1])) # for maxout
#     return [unif_invertible_layer_weights(k, m, n, slp, in_d, None, [-1., 1.], [0., 0.1])
#             for k, m, n, slp, in_d in zip(keys, in_sizes, out_sizes, [1]*len(layer_sizes), [False]*len(layer_sizes))]
#
#
# def mlp(params, inputs): # Maxout
#     z = inputs
#     for w, b in params[:-1]:
#         z = jnp.dot(z, w.T)+b
#         # z = nn.leaky_relu(z, lrelu_slope)
#         z1, z2 = jnp.split(z, 2, axis=-1)
#         z = jnp.maximum(z1, z2)
#     final_w, final_b = params[-1]
#     z = jnp.dot(z, final_w.T) + final_b
#     # concatenate xtm1 (IIA)
#     if jnp.ndim(inputs) == 1:
#         xtm1 = inputs[int(inputs.shape[0]/2):]
#     else:
#         xtm1 = inputs[:, int(inputs.shape[1]/2):]
#     z = jnp.concatenate([z, xtm1], axis=-1)
#     return z
