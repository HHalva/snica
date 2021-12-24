import jax
import jax.numpy as jnp
from jax import vjp, custom_vjp
from jax.tree_util import tree_reduce, tree_map
from jax.lax import while_loop
from functools import partial
from utils import tree_sub, tree_sum

@partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point(f, a, x_guess):
    def cond_fun(carry):
        x_prev, x, it = carry
        absdiff = tree_reduce(lambda c, t: c+jnp.sum(jnp.abs(t)), tree_sub(x_prev, x), .0)
        return absdiff > 1e-3

    def body_fun(carry):
        _, x, it = carry
        return x, f(a, x), it+1

    _, x_star, _ = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess), 0))
    return x_star


def fixed_point_fwd(f, a, x_init):
    x_star = fixed_point(f, a, x_init)
    return x_star, (a, x_star)


def fixed_point_rev(f, res, x_star_bar):
    a, x_star = res
    _, vjp_a = vjp(lambda a: f(a, x_star), a)
    a_bar, = vjp_a(fixed_point(partial(rev_iter, f),
                               (a, x_star, x_star_bar),
                               x_star_bar))
    return a_bar, tree_map(jnp.zeros_like, x_star)


def rev_iter(f, packed, u):
    a, x_star, x_star_bar = packed
    _, vjp_x = vjp(lambda x: f(a, x), x_star)
    return tree_sum([x_star_bar, vjp_x(u)[0]])


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)