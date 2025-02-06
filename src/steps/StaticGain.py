import jax
from functools import partial
from src.steps.Step import Step

class StaticGain(Step):

    def __init__(self, name, params):
        super().__init__(name, params)

    @partial(jax.jit, static_argnames=['self'])
    def compute_static(self, input_mat):
        return input_mat * self._params["factor"]