import jax
from functools import partial
from src.steps.Step import Step
from src import util
import jax.numpy as jnp

class ExampleStaticStep(Step):

    def __init__(self, name, params):
        mandatory_params = []
        super().__init__(name, params, mandatory_params)

        # Create addition inputs/outputs
        self.register_input("second_input")
        self.register_output("second_output")

    # JIT compile the compute function
    @partial(jax.jit, static_argnames=['self'])
    def compute(self, input_mats, **kwargs):
        input1 = input_mats[util.DEFAULT_INPUT_SLOT]
        input2 = input_mats["second_input"]

        output1 = input1 * 2 + input2 + 0.5
        output2 = 2 * jnp.square(jnp.abs(input2 - input1 + jnp.sin(jnp.transpose(input2) * 10)))
        
        return {util.DEFAULT_OUTPUT_SLOT: output1, 
                "second_output": output2}