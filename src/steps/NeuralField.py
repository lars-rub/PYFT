from src.steps.Step import Step
from src import util
from src import util_jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
from src.sigmoids import AbsSigmoid

# TODO move this somewhere else?
@jax.jit
def eulerStep_global(passedTime, input_mat, u_activation, prng_key, resting_level, global_inhibition, beta, theta, lateral_kernel_convolution_kernel, tau, input_noise_gain):
    sigmoided_u = AbsSigmoid(u_activation, beta, theta)
    lateral_interaction = jsp.signal.convolve(sigmoided_u, lateral_kernel_convolution_kernel, mode="same")

    sum_sigmoided_u = jnp.sum(sigmoided_u)

    d_u = -u_activation + resting_level + lateral_interaction + global_inhibition * sum_sigmoided_u + input_mat

    input_noise = jax.random.normal(prng_key, input_mat.shape)
    u_activation += (passedTime / tau) * d_u + ((jnp.sqrt(passedTime * 1000) / tau) / 1000) * input_noise_gain * input_noise
    # TODO ((jnp.sqrt(passedTime * 1000) / tau) / 1000)  ==  1000 * jnp.sqrt(passedTime) / tau    right? Try this out

    sigmoided_u = AbsSigmoid(u_activation, beta, theta)
    
    return sigmoided_u, u_activation

@partial(jax.jit, static_argnames=['resting_level', 'global_inhibition', 'beta', 'theta', 'tau', 'input_noise_gain'])
def eulerStep_global_partial(passedTime, input_mat, u_activation, prng_key, resting_level, global_inhibition, beta, theta, lateral_kernel_convolution_kernel, tau, input_noise_gain):
    sigmoided_u = AbsSigmoid(u_activation, beta, theta)
    lateral_interaction = jsp.signal.convolve(sigmoided_u, lateral_kernel_convolution_kernel, mode="same")

    sum_sigmoided_u = jnp.sum(sigmoided_u)

    d_u = -u_activation + resting_level + lateral_interaction + global_inhibition * sum_sigmoided_u + input_mat

    input_noise = jax.random.normal(prng_key, input_mat.shape)
    u_activation += (passedTime / tau) * d_u + ((jnp.sqrt(passedTime * 1000) / tau) / 1000) * input_noise_gain * input_noise
    # TODO ((jnp.sqrt(passedTime * 1000) / tau) / 1000)  ==  1000 * jnp.sqrt(passedTime) / tau    right? Try this out

    sigmoided_u = AbsSigmoid(u_activation, beta, theta)
    
    return sigmoided_u, u_activation

parallel_euler_step = jax.vmap(eulerStep_global)

class NeuralField(Step):

    def __init__(self, name, params):
        super().__init__(name, params)
        self.is_dynamic = True
        self.needs_input_connections = False
        self._device_idx = util_jax.next_gpu()
        self._max_incoming_connections = jnp.inf
        self._euler_func = eulerStep_global_partial if util_jax.cfg['euler_step_partial'] else eulerStep_global
        self.reset()
    
    def compute_static(self, input_mat):
        return self._output_buf

    #@line_profiler.profile
    def compute_dynamic(self, passedTime, input_mat, prng_key):
        sigmoided_u, u = self._euler_func(passedTime, input_mat, self._buf["activation"], prng_key, self._params["resting_level"], self._params["global_inhibition"],
                                          self._params["sigmoid"]._beta, self._params["sigmoid"]._theta, self._params["lateral_kernel_convolution"].get_kernel(), self._params["tau"], self._params["input_noise_gain"])
        sigmoided_u.block_until_ready()
        u.block_until_ready()
        self._buf["activation"] = u
        self._output_buf = sigmoided_u
        return sigmoided_u
    
    def reset(self): # Override
        self._buf["activation"] = util_jax.ones(self._params["shape"]) * self._params["resting_level"]
        self._output_buf = util_jax.zeros(self._params["shape"], device_idx=self._device_idx)
