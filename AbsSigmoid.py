import jax.numpy as jnp
from functools import partial
import jax

class AbsSigmoid: # TODO rethink if this needs to be a class or if the field can just have parameters beta/theta etc if there is only one sigmoid per field always anyways

    def __init__(self, beta, theta):
        self._beta = beta
        self._theta = theta # threshold

    @partial(jax.jit, static_argnames=['self'])
    def apply(self, x):
        raise Exception("This function is currently not used, a copy of this resides in NeuralField.py.")
        return 0.5 * (1.0 + self._beta * (x - self._theta) / (1.0 + self._beta * jnp.abs(x - self._theta)))
    