import numpy as np
import jax
from jax import jit
from functools import partial
from steps.Step import Step
from time_message import tprint

class StaticGain(Step):

    def __init__(self, name, params):
        super().__init__(name, params)

    #@partial(jax.jit, static_argnames=['self', 'arch'])
    def compute_static_old(self, arch):
        arch.check_compiled()
        input_list = arch.get_incoming_steps(self.get_name())
        num_incoming_steps = len(input_list)
        if num_incoming_steps == 0:
            raise ValueError(f"StaticGain {self.get_name()} has no incoming connection")

        input_mat = input_list[0].compute_static_old(arch)
        #print(f"Compute StaticGain {self._name}")
        return input_mat * self._params["factor"]
    

    @partial(jax.jit, static_argnames=['self'])
    def compute_static(self, input_mat):
        return input_mat * self._params["factor"]