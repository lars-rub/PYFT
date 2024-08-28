from steps.Step import Step
from time_message import tprint
import util
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
# import os
# os.environ["LINE_PROFILE"] = "0"
# import line_profiler


## TODO the sigmoid should not be hardcoded here, however this was more convenient for now to test vmap
@jax.jit
def sigmoid(x, beta, theta):
    return 0.5 * (1.0 + beta * (x - theta) / (1.0 + beta * jnp.abs(x - theta)))

# TODO can we use static_argnames for arguments like resting_level, global_inhibition, etc.?
# TODO move this somewhere else?
@jax.jit
def eulerStep_global(passedTime, input_mat, u_activation, prng_key, resting_level, global_inhibition, beta, theta, lateral_kernel_convolution_kernel, tau, input_noise_gain):
    sigmoided_u = sigmoid(u_activation, beta, theta)
    lateral_interaction = jsp.signal.convolve2d(sigmoided_u, lateral_kernel_convolution_kernel, mode="same")

    sum_sigmoided_u = jnp.sum(sigmoided_u)

    d_u = -u_activation + resting_level + lateral_interaction + global_inhibition * sum_sigmoided_u + input_mat

    input_noise = jax.random.normal(prng_key, input_mat.shape)

    u_activation += (passedTime / tau) * d_u + ((jnp.sqrt(passedTime * 1000) / tau) / 1000) * input_noise_gain * input_noise

    # TODO ((jnp.sqrt(passedTime * 1000) / tau) / 1000)  ==  1000 * jnp.sqrt(passedTime) / tau    right? Try this out
    
    return sigmoided_u, u_activation

parallel_euler_step = jax.vmap(eulerStep_global)

class NeuralField(Step):

    def __init__(self, name, params):
        super().__init__(name, params)
        self.is_dynamic = True
        self._device_idx = util.next_gpu()
        self._max_incoming_connections = jnp.inf
        self.reset()

    
    @partial(jax.jit, static_argnames=['self'])
    def eulerStep_not_mappable(self, passedTime, input_mat, u_activation, prng_key):

        resting_level = self._params["resting_level"]
        global_inhibition = self._params["global_inhibition"]

        sigmoided_u = self._params["sigmoid"].apply(u_activation)
        lateral_interaction = jsp.signal.convolve2d(sigmoided_u, self._params["lateral_kernel_convolution"].get_kernel(), mode="same")
        
        sum_sigmoided_u = jnp.sum(sigmoided_u)

        d_u = -u_activation + resting_level + lateral_interaction + global_inhibition * sum_sigmoided_u + input_mat

        tau = self._params["tau"]
        input_noise_gain = self._params["input_noise_gain"]
        input_noise = jax.random.normal(prng_key, input_mat.shape)

        u_activation += (passedTime / tau) * d_u + ((jnp.sqrt(passedTime * 1000) / tau) / 1000) * input_noise_gain * input_noise
        return sigmoided_u, u_activation
    
    def compute_static_old(self, arch):
        return self._output_buf
    
    def compute_static(self, input_mat):
        return self._output_buf

    #@line_profiler.profile
    def compute_dynamic(self, passedTime, input_mat, prng_key):
        raise Exception("This functions is not used anymore since the introduction of vmap. Maybe reintroduce it in some way?")
        #with jax.profiler.trace("tensorboard_log/tmp"):
        sigmoided_u, u = self.eulerStep(passedTime, input_mat, self._activation_buffer, prng_key)
        sigmoided_u.block_until_ready()
        u.block_until_ready()
        self._activation_buffer = u
        return sigmoided_u
    
    def update_input_old(self, arch):
        input_sum = None
        incoming_steps = arch.get_incoming_steps(self.get_name())
        if len(incoming_steps) == 0:
            input_sum = util.zeros(self._params["shape"], device_idx=self._device_idx)
        else:
            for step in incoming_steps:
                result = step.compute_static_old(arch)
                if input_sum is None:
                    input_sum = result
                else:
                    input_sum += result
        self._input = input_sum
        #print(f"Update NeuralField input {self._name} to {input_sum}")

    def update_input(self, arch):
        input_sum = None
        incoming_steps = arch.get_incoming_steps(self.get_name())
        if len(incoming_steps) == 0:
            input_sum = util.zeros(self._params["shape"], device_idx=self._device_idx)
        else:
            for step in incoming_steps:
                result = step.get_output_buffer()
                if input_sum is None:
                    input_sum = result
                else:
                    input_sum += result
        #print(f"Update NeuralField input {self._name} to {input_sum}")
        return input_sum
    
    def reset(self): # Override
        self._input = None # TODO remove (belongs to _old functions)
        self._activation_buffer = util.ones(self._params["shape"]) * self._params["resting_level"]
        self._output_buf = util.zeros(self._params["shape"], device_idx=self._device_idx)

    def get_input_old(self):
        return self._input
    