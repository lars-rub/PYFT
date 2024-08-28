import jax.numpy as jnp
from steps.Step import Step
import jax
import util
from functools import partial
from time_message import tprint

class GaussInput(Step):

    def __init__(self, name, params):
        super().__init__(name, params)
        if len(params["shape"]) != 2 or params["shape"][0] != params["shape"][1]:
            raise ValueError(f"GaussInput {name} requires square 2D shape (currently)")
        self._side_length = params["shape"][0]
        self._kernel = self.gkern()

    def gkern(self):
        # creates gaussian kernel with specified side length and sigma
        ax = jnp.linspace(-(self._side_length - 1) / 2., (self._side_length - 1) / 2., self._side_length, dtype=util.cfg["jdtype"])
        gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(self._params["sigma"]))#, dtype=util.cfg["jdtype"])
        kernel = jnp.outer(gauss, gauss)
        return self._params["amplitude"] * kernel# / np.sum(kernel)
    
    @partial(jax.jit, static_argnames=['self', 'arch'])
    def compute_static_old(self, arch):
        arch.check_compiled()
        return self._kernel
    
    @partial(jax.jit, static_argnames=['self'])
    def compute_static(self, input_mat): # TODO different handling of sources? Doesnt really need the input_mat argument
        return self._kernel