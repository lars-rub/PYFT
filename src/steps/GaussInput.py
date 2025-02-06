import jax.numpy as jnp
from src.steps.Step import Step
import jax
from src import util_jax
from functools import partial

class GaussInput(Step):

    def __init__(self, name, params):
        super().__init__(name, params)
        self._dimensionality = len(params["shape"])
        for size in params["shape"][1:]:
            if size != params["shape"][0]:
                raise ValueError(f"GaussInput {name} requires equal shape sizes for all dimensions")
        self.is_source = True
        self._side_length = params["shape"][0]
        self._kernel = self.gkern()

    def gkern(self):
        # creates gaussian kernel with specified side length and sigma
        ax = jnp.linspace(-(self._side_length - 1) / 2., (self._side_length - 1) / 2., self._side_length, dtype=util_jax.cfg["jdtype"])
        gauss_1d = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(self._params["sigma"]))
        kernel = gauss_1d
        for dim in range(2, self._dimensionality + 1):
            kernel = jnp.outer(kernel, gauss_1d).reshape((self._side_length,) * (dim))
        return self._params["amplitude"] * kernel
    
    @partial(jax.jit, static_argnames=['self'])
    def compute_static(self, input_mat): # TODO different handling of sources? Doesnt really need the input_mat argument
        return self._kernel