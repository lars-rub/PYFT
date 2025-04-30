import jax
from functools import partial
from src.steps.Step import Step
from src import util

class StaticGain(Step):

    def __init__(self, name, params):
        mandatory_params = ["factor"]
        super().__init__(name, params, mandatory_params)

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input = input_mats[util.DEFAULT_INPUT_SLOT]

        # TODO Add factor as an argument an make this a non-member function, so only one StaticGain::compute() function needs to be compiled
        output = input * self._params["factor"]
        
        return {util.DEFAULT_OUTPUT_SLOT: output}