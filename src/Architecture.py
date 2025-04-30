from src import util
from src import util_jax
import jax
from functools import partial
import time
import numpy as np
from src.steps.NeuralField import NeuralField
from src.steps.ExampleDynamicStep import ExampleDynamicStep

class Architecture:
    def __init__(self):
        self.element_map = {}
        self.connection_map_reversed = {}
        self.compiled = False
        util.set_architecture(self)

    def is_compiled(self):
        return self.compiled

    @partial(jax.jit, static_argnames=['self'])
    def check_compiled(self):
        if not self.is_compiled():
            raise util.ArchitectureNotCompiledException
    
    def check_not_compiled(self):
        if self.is_compiled():
            raise util.ArchitectureCompiledException

    def construct_static_compilation_graph(self, permutation): # TODO remove permutation?
        self.check_not_compiled()
        compiled_steps = []
        compilation_graph_static = []
        dynamic_steps_permuted = [self.dynamic_steps_c[i] for i in permutation]
        # iterate through all dynamic steps and create a subgraph of the tree of incoming static steps to this dynamic step.
        for dynamic_step in dynamic_steps_permuted:
            subgraph = []
            incoming_steps = [step.split(".")[0] for step in self.get_incoming_steps(dynamic_step.get_name())]
            if len(incoming_steps) == 0:
                continue
            bfs_queue = incoming_steps
            # Do a backwards BFS to find the subtree
            while len(bfs_queue) > 0:
                current = bfs_queue.pop(0)
                if current in compiled_steps or self.element_map[current].is_dynamic:
                    continue
                compiled_steps.append(current)
                incoming_steps = [step.split(".")[0] for step in self.get_incoming_steps(current)]
                subgraph.append([current, incoming_steps]) # TODO remove incoming_steps?
                bfs_queue += incoming_steps
            subgraph = subgraph[::-1]
            compilation_graph_static += subgraph
        self.compilation_graph_static_c = compilation_graph_static
        print("\nStatic step compilation graph:\n" + "\n".join([f"{elem[0]:<8} <-- {str(elem[1])}" for elem in compilation_graph_static]) + "\n")

    # If warmup is set to true, the architecture is run once and then reset, effectively precompiling all JIT compilable functions in the architecture (e.g. euler step of fields)
    def compile(self, warmup=True):
        self.check_not_compiled()

        ## Compile
        self.cfg_c = util_jax.cfg
        self.dynamic_steps_c = [element for element in self.element_map.values() if element.is_dynamic]
        # Compute the compilation graph of static steps which determines the order in which they need to be executed.
        self.construct_static_compilation_graph(range(len(self.dynamic_steps_c)))

        # Precompile static steps
        for graph_elem in self.compilation_graph_static_c:
            step_name, incoming_steps = graph_elem
            step = self.get_element(step_name) # TODO put step in graph_elem to avoid this call?
            step.pre_compile(self)

        ## Warmup
        self.compiled = True
        self.check_compiled()
        if warmup:
            self.tick()
            self.reset_steps()

    def reset_steps(self):
        self.check_compiled()
        for element in self.element_map.values():
            element.reset()

    def add_element(self, element):
        self.check_not_compiled()
        name = element.get_name()
        if name in self.element_map:
            raise Exception(f"Architecture::add_element(): Element {name} already exists in Architecture")
        self.element_map[name] = element

        for slot in element.input_slot_names: # TODO improve?
            self.connection_map_reversed[name + "." + slot] = []

    def __add__(self, element):
        self.add_element(element)
        return self

    def get_elements(self):
        return self.element_map
    
    def get_element(self, name):
        if name not in self.element_map:
            raise Exception(f"Architecture::get_element(): Element {name} not found in Architecture")
        return self.element_map[name]
    
    # source and dest are strings of the form "step_name.slot_name" or "step_name" which will use the first slot (util.DEFAULT_*_SLOT)
    def connect_to(self, source, dest):
        self.check_not_compiled()
        if not isinstance(source, str) or not isinstance(dest, str):
            raise Exception(f"Architecture::connect_to(): source and dest must be strings, but got {type(source)} and {type(dest)}. (TODO: add support for Step and Slot type as argument)")
        # Set default slot if not specified
        if not "." in source:
            source = source + "." + util.DEFAULT_OUTPUT_SLOT
        if not "." in dest:
            dest = dest + "." + util.DEFAULT_INPUT_SLOT
        
        source_step = source.split(".")[0]
        dest_step = dest.split(".")[0]
        dest_slot = dest.split(".")[1]
        
        for name in [source_step, dest_step]:
            if name not in self.element_map:
                raise Exception(f"Architecture::connect_to(): Element {name} not found in Architecture")
        if source_step == dest_step:
            raise Exception(f"Architecture::connect_to(): Cannot connect element {source} to itself")
        if source in self.connection_map_reversed[dest]:
            raise Exception(f"Architecture::connect_to(): Connection from {source} to {dest} already exists")
        
        if len(self.connection_map_reversed[dest]) >= self.get_element(dest_step).get_max_incoming_connections(dest_slot):
            raise Exception(f"Architecture::connect_to(): Element {dest} already has {self.get_element(dest_step).get_max_incoming_connections(dest_slot)} incoming connection(s)")

        self.connection_map_reversed[dest].append(source)
    
    # dest is a string of the form "step_name.slot_name" or "step_name" which will return all incoming steps to that step
    def get_incoming_steps(self, dest):
        incoming_steps = []
        if "." in dest:
            # If slot is specified, return all incoming steps to that slot
            incoming_steps = self.connection_map_reversed[dest]
        else:
            # If no slot is specified, return all incoming steps to all slots
            all_slots = self.get_element(dest).input_slot_names
            for slot in all_slots:
                incoming_steps += self.connection_map_reversed[dest + "." + slot]
        return incoming_steps

    def run_simulation(self, tick_func, steps_to_plot, num_steps, print_timing=True):
        history = []
        timing_all = []
        start_time = time.time()
        for _ in range(num_steps):

            # Execute tick function
            timing_all.append(tick_func())

            # Save output of steps to plot
            if len(steps_to_plot) > 0:
                data = []
                for to_plot in steps_to_plot:
                    step, slot = to_plot.split(".") if "." in to_plot else [to_plot, util.DEFAULT_OUTPUT_SLOT]
                    data.append(self.get_element(step).get_buffer(slot))
                history.append(data)

        end_time = time.time()
        timing = np.mean(timing_all, axis=0)
        ms_per_tick = 1000 * (end_time - start_time) / num_steps
        if print_timing:
            print(f"{ms_per_tick:6.2f} ms / time step")
            print(f"{(end_time - start_time):6.2f} s total duration\n")
            print(f"{(1000 * timing[0]):6.2f} ms average time for computation of static steps")
            print(f"{(1000 * timing[1]):6.2f} ms average time for dynamic computation")
        return history, ms_per_tick, timing

    def tick(self):
        self.check_compiled()
        start_time = time.time()

        ## -- Update static steps --

        for graph_elem in self.compilation_graph_static_c:
            step_name, incoming_steps = graph_elem
            step = self.get_element(step_name) # TODO put step in graph_elem to avoid this call?

            if step.is_source:
                input_sum = None
            else:
                input_sum = step.update_input(self)
            
            step.buffer = step.compute(input_sum)
        static_update_time = time.time() - start_time

        ## -- Update dynamic steps --

        delta_t = self.cfg_c["delta_t"]
        random_keys = util_jax.next_random_keys(len(self.dynamic_steps_c))
        dynamic_output = []

        # Run compute on dynamic steps
        for i, step in enumerate(self.dynamic_steps_c):
            input_mats = step.update_input(self)
            output = step.compute(input_mats, prng_key=random_keys[i], delta_t=delta_t)
            dynamic_output.append(output)

        # Block output and save to buffers *after* all executions are started to allow jax to parallelize compute calls
        for i, step in enumerate(self.dynamic_steps_c):
            step.post_compute(dynamic_output[i])
        dynamic_update_time = time.time() - (start_time + static_update_time)
        return static_update_time, dynamic_update_time
    