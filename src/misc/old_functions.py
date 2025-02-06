import jax.numpy as jnp
import jax
from src import util_jax
from functools import partial
from src.steps.Step import Step
import jax.scipy as jsp

# When jit compilation is attempted, jax recompiles for every field even if the parameters are the same as self is always different => Leads to huge compilation times
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

    # def update_input_old(self, arch):
    #     input_sum = None
    #     incoming_steps = arch.get_incoming_steps(self.get_name())
    #     if len(incoming_steps) == 0:
    #         input_sum = util_jax.zeros(self._params["shape"], device_idx=self._device_idx)
    #     else:
    #         for step in incoming_steps:
    #             result = step.compute_static_old(arch)
    #             if input_sum is None:
    #                 input_sum = result
    #             else:
    #                 input_sum += result
    #     self._input = input_sum
    #     #print(f"Update NeuralField input {self._name} to {input_sum}")

    # def update_input(self, arch):
    #     input_sum = None
    #     incoming_steps = arch.get_incoming_steps(self.get_name())
    #     if len(incoming_steps) == 0:
    #         input_sum = util_jax.zeros(self._params["shape"], device_idx=self._device_idx)
    #     else:
    #         for step in incoming_steps:
    #             result = step.get_output_buffer()
    #             if input_sum is None:
    #                 input_sum = result
    #             else:
    #                 input_sum += result
    #     #print(f"Update NeuralField input {self._name} to {input_sum}")
    #     return input_sum
    
    # def tick_old_static_recursive(self):
    #     self.check_compiled() # TODO measure time, is jit compiled so should be fast but check
    #     start_time = time.time()
    #     delta_t = self.cfg_c["delta_t"] # "passed time since last tick (fixed value for 'simulated time')" in seconds

    #     # Update static steps
    #     for field in self.fields_list_c: # TODO parallelize with pmap?
    #         field.update_input_old(self)
    #     static_update_time = time.time() - start_time
    #     # TODO block here

    #     # Update Fields (eulerStep)
    #     for field in self.fields_list_c: # TODO parallelize with pmap?
    #         field.set_output_buffer(field.compute_dynamic(delta_t, field.get_input_old()))
    #     dynamic_update_time = time.time() - (start_time + static_update_time)

    #     return static_update_time, dynamic_update_time


class GaussKernel2D:

    def __init__(self, params):
        self._params = params
        self._sigma = params["sigma"]
        self._amplitude = params["amplitude"]
        self._side_length = self._estimate_size()[0]

    def _estimate_size(self):
        limit = 5
        widths = []
        for dim in range(1): # TODO make this whole thing multi dimensional
            # sigma = sigmas[dim]
            if self._sigma == 0:
                widths.append(1)
            else:
                width = int(jnp.ceil(limit * self._sigma))
                if width % 2 == 0:
                    width += 1
                widths.append(width)
        return widths

    @partial(jax.jit, static_argnames=['self'])
    def get_kernel(self):
        # creates gaussian kernel with specified side length and sigma
        ax = jnp.linspace(-(self._side_length - 1) / 2., (self._side_length - 1) / 2., self._side_length, dtype=util_jax.cfg["jdtype"])
        gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(self._sigma))
        kernel = jnp.outer(gauss, gauss)
        return self._amplitude * kernel# / np.sum(kernel)import jax.numpy as jnp


class GaussInput2D(Step):

    def __init__(self, name, params):
        super().__init__(name, params)
        if len(params["shape"]) != 2 or params["shape"][0] != params["shape"][1]:
            raise ValueError(f"GaussInput {name} requires square 2D shape (currently)")
        self.is_source = True
        self._side_length = params["shape"][0]
        self._kernel = self.gkern()

    def gkern(self):
        # creates gaussian kernel with specified side length and sigma
        ax = jnp.linspace(-(self._side_length - 1) / 2., (self._side_length - 1) / 2., self._side_length, dtype=util_jax.cfg["jdtype"])
        gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(self._params["sigma"]))#, dtype=util_jax.cfg["jdtype"])
        kernel = jnp.outer(gauss, gauss)
        return self._params["amplitude"] * kernel# / np.sum(kernel)
    
    @partial(jax.jit, static_argnames=['self'])
    def compute_static(self, input_mat): # TODO different handling of sources? Doesnt really need the input_mat argument
        return self._kernel