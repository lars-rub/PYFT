import jax.numpy as jnp
import jax
import util
from functools import partial

class GaussKernel:

    def __init__(self, params):
        self._params = params
        self._sigma = params["sigma"]
        self._side_length = params["shape"][0]
        self._amplitude = params["amplitude"]

    @partial(jax.jit, static_argnames=['self'])
    def get_kernel(self):
        # creates gaussian kernel with specified side length and sigma
        ax = jnp.linspace(-(self._side_length - 1) / 2., (self._side_length - 1) / 2., self._side_length, dtype=util.cfg["jdtype"])
        gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(self._sigma))
        kernel = jnp.outer(gauss, gauss)
        return self._amplitude * kernel# / np.sum(kernel)