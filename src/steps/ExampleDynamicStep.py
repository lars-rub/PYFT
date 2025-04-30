from src.steps.Step import Step
from src import util
from src import util_jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial

class ExampleDynamicStep(Step):

    def __init__(self, name, params):
        mandatory_params = ["shape"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = jnp.inf

    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        if not "prng_key" in kwargs:
            raise Exception("prng_key is a mandatory kwarg to dynamic compute()")
        
        # Input
        input_mat = input_mats[util.DEFAULT_INPUT_SLOT]
        prng_key = kwargs["prng_key"]
        
        # Computation
        input_noise = jax.random.normal(prng_key, input_mat.shape)
        output = input_mat + jnp.abs(input_noise) * 0.5

        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: output}
    
