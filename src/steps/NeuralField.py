from src.steps.Step import Step
from src import util
from src import util_jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
from src.sigmoids import AbsSigmoid

# This singleton construct is needed as we need to specify the static_argnames in the compiler directive depending on the user input
euler_func = None
def euler_func_singleton(static):
    global euler_func
    if euler_func is not None:
        return euler_func

    static_argnames = []
    if static:
        static_argnames = ['resting_level', 'global_inhibition', 'beta', 'theta', 'tau', 'input_noise_gain']

    # euler step computation
    @partial(jax.jit, static_argnames=static_argnames)
    def eulerStep(passedTime, input_mat, u_activation, prng_key, resting_level, global_inhibition, beta, theta, lateral_kernel_convolution_kernel, tau, input_noise_gain):
        sigmoided_u = AbsSigmoid(u_activation, beta, theta) # Could be optimized, we don't need this sigmoid computation if we pass the value of the output buffer to this function (which effectively is the sigmoided_u)
        lateral_interaction = jsp.signal.convolve(sigmoided_u, lateral_kernel_convolution_kernel, mode="same")

        sum_sigmoided_u = jnp.sum(sigmoided_u)

        d_u = -u_activation + resting_level + lateral_interaction + global_inhibition * sum_sigmoided_u + input_mat

        input_noise = jax.random.normal(prng_key, input_mat.shape)
        u_activation += (passedTime / tau) * d_u + ((jnp.sqrt(passedTime * 1000) / tau) / 1000) * input_noise_gain * input_noise

        sigmoided_u = AbsSigmoid(u_activation, beta, theta)
        
        return sigmoided_u, u_activation
    euler_func = eulerStep
    return eulerStep

class NeuralField(Step):

    def __init__(self, name, params):
        mandatory_params = ["shape", "sigmoid", "resting_level", "global_inhibition", "input_noise_gain", "tau", "lateral_kernel_convolution"]
        super().__init__(name, params, mandatory_params, is_dynamic=True)
        self.needs_input_connections = False
        self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = jnp.inf
        self._delta_t = util_jax.get_config()["delta_t"]
        self._euler_func =  euler_func_singleton(util_jax.cfg['euler_step_static_precompile'])
        self._lateral_kernel = self._params["lateral_kernel_convolution"].get_kernel()

        for dim in range(len(self._params["shape"])):
            if self._params["shape"][dim] < self._lateral_kernel.shape[dim]:
                raise ValueError(f"NeuralField {name} requires shape {self._params['shape']} to be larger than lateral kernel "\
                                 f"shape {self._lateral_kernel.shape} in every dimension. Reduce lateral kernel sigma.")
        self.reset()

    # required kwargs are: delta_t, prng_key
    def compute(self, input_mats, **kwargs):
        if not "prng_key" in kwargs:
            raise Exception("prng_key is a mandatory kwarg to dynamic compute()")
        input_mat = input_mats[util.DEFAULT_INPUT_SLOT]

        # Call the euler function with the input_mat and all parameters
        sigmoided_u, u = self._euler_func(self._delta_t, input_mat, self.buffer["activation"], kwargs["prng_key"], self._params["resting_level"], self._params["global_inhibition"],
                                          self._params["sigmoid"]._beta, self._params["sigmoid"]._theta, self._lateral_kernel, self._params["tau"], self._params["input_noise_gain"])
        
        # Return output
        return {util.DEFAULT_OUTPUT_SLOT: sigmoided_u, 
                "activation": u}
    
    def reset(self): # Override
        self.buffer["activation"] = util_jax.ones(self._params["shape"]) * self._params["resting_level"]
        self.reset_buffer(util.DEFAULT_OUTPUT_SLOT)
