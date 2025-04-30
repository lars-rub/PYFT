import jax.numpy as jnp
from functools import partial
import jax

class AbsSigmoid: # TODO rethink if this needs to be a class or if the field can just have parameters beta/theta etc if there is only one sigmoid per field always anyways, or if we just use src.sigmoids

    def __init__(self, beta, theta):
        self._beta = beta
        self._theta = theta # threshold
