import jax.numpy as jnp
import jax
from src import util_jax
from functools import partial

class GaussKernel:

    def __init__(self, params):
        self._params = params
        self._dimensionality = len(params["sigma"])
        self.is_source = True
        self._side_length = self._estimate_size()[0]
        self._kernel = self.gkern()

    def _estimate_size(self):
        limit = 5
        widths = []
        for dim in range(1): # TODO make this whole thing multi dimensional
            sigma = self._params["sigma"][dim]
            if sigma == 0:
                widths.append(1)
            else:
                width = int(jnp.ceil(limit * sigma))
                if width % 2 == 0:
                    width += 1
                widths.append(width)
        return widths

    def gkern(self):
        # creates gaussian kernel with specified side length and sigma
        ax = jnp.linspace(-(self._side_length - 1) / 2., (self._side_length - 1) / 2., self._side_length, dtype=util_jax.cfg["jdtype"])
        gauss_1d = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(self._params["sigma"][0]))
        kernel = gauss_1d
        for dim in range(2, self._dimensionality + 1):
            kernel = jnp.outer(kernel, gauss_1d).reshape((self._side_length,) * (dim))
        return self._params["amplitude"] * kernel
    
    @partial(jax.jit, static_argnames=['self'])
    def get_kernel(self):
        return self._kernel