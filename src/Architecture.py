from src.steps.NeuralField import parallel_euler_step
from src import util
from src import util_jax
import jax
import jax.numpy as jnp
from functools import partial
import time
import numpy as np

# import os
# os.environ["LINE_PROFILE"] = "0"
# import line_profiler

class Architecture:
    def __init__(self):
        self.element_map = {}
        self.connection_map = {}
        self.connection_map_reversed = {}
        self.compiled = False

    def is_compiled(self):
        return self.compiled

    @partial(jax.jit, static_argnames=['self'])
    def check_compiled(self):
        if not self.is_compiled():
            raise util.ArchitectureNotCompiledException
    
    def check_not_compiled(self):
        if self.is_compiled():
            raise util.ArchitectureCompiledException

    def construct_static_compilation_graph(self, field_permutation):
        self.check_not_compiled()
        compiled_steps = []
        compilation_graph_static = []
        fields_permuted = [self.fields_list_c[i] for i in field_permutation]
        for field in fields_permuted:
            subgraph = []
            if not field.get_name() in self.connection_map_reversed:
                continue
            bfs_queue = self.connection_map_reversed[field.get_name()].copy()
            while len(bfs_queue) > 0:
                current = bfs_queue.pop(0)
                if current in compiled_steps or self.element_map[current].is_dynamic:
                    continue
                compiled_steps.append(current)
                subgraph.append([current, self.get_incoming_steps(current)])
                bfs_queue += self.connection_map_reversed[current]
            subgraph = subgraph[::-1]
            compilation_graph_static += subgraph
        self.compilation_graph_static_c = compilation_graph_static
        print("\nStatic step compilation graph:\n" + "\n".join([f"{elem[0]:<5} {str([a.get_name() for a in elem[1]])}" for elem in compilation_graph_static]) + "\n")

    def compile(self, warmup=True):
        self.check_not_compiled()
        self.cfg_c = util_jax.cfg
        self.fields_list_c = self.get_fields()
        self.construct_static_compilation_graph(range(len(self.fields_list_c)))
        
        if self.cfg_c["use_vmap"]:
            # TODO this will be reworked, but this general concept is required for vmap to be somehow efficient
            self.resting_levels_c       = jnp.array([field._params["resting_level"] for field in self.fields_list_c])
            self.global_inhibitions_c   = jnp.array([field._params["global_inhibition"] for field in self.fields_list_c])
            self.lateral_kerns_c        = jnp.array([field._params["lateral_kernel_convolution"].get_kernel() for field in self.fields_list_c])
            self.input_noise_gains_c    = jnp.array([field._params["input_noise_gain"] for field in self.fields_list_c])
            self.taus_c                 = jnp.array([field._params["tau"] for field in self.fields_list_c])
            self.delta_ts_c             = jnp.repeat(self.cfg_c["delta_t"], len(self.fields_list_c))
            self.betas_c                = jnp.array([field._params["sigmoid"]._beta for field in self.fields_list_c])
            self.thetas_c               = jnp.array([field._params["sigmoid"]._theta for field in self.fields_list_c])
            self.activation_buffers_c_dynamic = jnp.array([util_jax.ones(field._params["shape"]) * field._params["resting_level"] for field in self.fields_list_c])

        self.compiled = True
        self.check_compiled()
        if warmup:
            self.tick()
            self.reset_steps()

    def reset_steps(self):
        for element in self.element_map.values():
            element.reset()
        self.activation_buffers_c_dynamic = jnp.array([util_jax.ones(field._params["shape"]) * field._params["resting_level"] for field in self.fields_list_c])

    def add_element(self, element):
        self.check_not_compiled()
        name = element.get_name()
        if name in self.element_map:
            raise Exception(f"Architecture::add_element(): Element {name} already exists in Architecture")
        self.element_map[name] = element

        self.connection_map[name] = []
        self.connection_map_reversed[name] = []

    def get_elements(self):
        return self.element_map
    
    def get_element(self, name):
        if name not in self.element_map:
            raise Exception(f"Architecture::get_element(): Element {name} not found in Architecture")
        return self.element_map[name]

    def get_fields(self):
        if self.compiled:
            raise Exception("Architecture::get_fields(): For compiled architectures use the 'fields_list_c' attribute instead of the 'get_fields()' method")
        return [element for element in self.element_map.values() if element.is_dynamic]
    
    def connect_to(self, source, dest):
        self.check_not_compiled()
        for name in [source, dest]:
            if name not in self.element_map:
                raise Exception(f"Architecture::connect_to(): Element {name} not found in Architecture")
        if source == dest:
            raise Exception(f"Architecture::connect_to(): Cannot connect element {source} to itself")
        if dest in self.connection_map[source]:
            raise Exception(f"Architecture::connect_to(): Connection from {source} to {dest} already exists")
        
        if len(self.connection_map_reversed[dest]) >= self.get_element(dest).get_max_incoming_connections():
            raise Exception(f"Architecture::connect_to(): Element {dest} already has {self.get_element(dest).get_max_incoming_connections()} incoming connection(s)")

        self.connection_map[source].append(dest)
        self.connection_map_reversed[dest].append(source)
    
    def get_incoming_steps(self, dest_name):
        return [self.element_map[name] for name in self.connection_map_reversed[dest_name]]

    def run_simulation(self, tick_func, steps_to_plot, num_steps, print_timing=True):
        history = []
        start_time = time.time()
        timing_all = []
        for _ in range(num_steps):
            # Execute tick function
            timing_all.append(tick_func())
            # Save output of steps to plot
            if len(steps_to_plot) > 0:
                history.append([self.get_element(step.split(".")[0]).get_buffer(step.split(".")[1]) if "." in step else self.get_element(step).get_output_buffer() for step in steps_to_plot])
        end_time = time.time()
        timing = np.mean(timing_all, axis=0)
        ms_per_tick = 1000 * (end_time - start_time) / num_steps
        if print_timing:
            print(f"{ms_per_tick:6.2f} ms / time step")
            print(f"{(end_time - start_time):6.2f} s total duration")
            print("")
            print(f"{(1000 * timing[0]):6.2f} ms average time for computation regarding static steps")
            print(f"{(1000 * timing[1]):6.2f} ms average time for dynamic computation (without eulerStep)")
            print(f"{(1000 * timing[2]):6.2f} ms average time for eulerStep: ")
        return history, ms_per_tick

    #@line_profiler.profile   # !! Important: if you want to use line_profiler for this function, you should somehow disable it for the very first call of this function (which is executed in Architecture::compile())
    def tick(self):
        self.check_compiled()
        start_time = time.time()

        # Update static steps
        for graph_elem in self.compilation_graph_static_c:
            step_name, incoming_steps = graph_elem
            step = self.get_element(step_name) # TODO put step in graph_elem to avoid this call?

            if step.is_source:
                input_sum = None
            else:
                input_sum = step.update_input(self)

            step.set_output_buffer(step.compute_static(input_sum))

        static_update_time = time.time() - start_time
        # TODO block here

        if self.cfg_c["use_vmap"]:
            in_mats = []
            # Update Fields (eulerStep)
            for field in self.fields_list_c: # TODO parallelize?
                in_mat = field.update_input(self)
                if util_jax.use_multiple_devices:
                    in_mat = jax.device_put(in_mat, jax.local_devices()[field._device_idx]) # TODO check if it is already there?
                in_mats.append(in_mat)
            in_mats = jnp.stack(in_mats, axis=0)

            # eulerSteps
            #activation_buffers = jnp.stack([field.get_buffer("activation") for field in self.fields_list_c], axis=0)
            random_keys = util_jax.next_random_keys(len(self.fields_list_c))
            #sigmoided_us = jnp.stack([field._params["sigmoid"].apply(field..get_buffer("activation")) for field in self.fields_list_c], axis=0)

            real_dynamic_start = time.time()
            sigmoided_us, us = parallel_euler_step(self.delta_ts_c, in_mats, self.activation_buffers_c_dynamic, random_keys, self.resting_levels_c, self.global_inhibitions_c, self.betas_c, self.thetas_c, self.lateral_kerns_c, self.taus_c, self.input_noise_gains_c)
            sigmoided_us.block_until_ready()
            us.block_until_ready()
            real_dynamic_time = time.time() - real_dynamic_start

            self.activation_buffers_c_dynamic = us
            for i, field in enumerate(self.fields_list_c):
                field._output_buf = sigmoided_us[i]
                #field._buf["activation"] = us[i]

            dynamic_update_time = time.time() - (start_time + static_update_time + real_dynamic_time)
        else:
            delta_t = self.cfg_c["delta_t"]
            # run compute_dynamic(self, passedTime, input_mat, prng_key) for every field
            random_keys = util_jax.next_random_keys(len(self.fields_list_c))
            for i, field in enumerate(self.fields_list_c):
                prng_key = random_keys[i]
                input_mat = field.update_input(self)
                field.compute_dynamic(delta_t, input_mat, prng_key)
            dynamic_update_time = time.time() - (start_time + static_update_time)
            real_dynamic_time = 0
        return static_update_time, dynamic_update_time, real_dynamic_time
    