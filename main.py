import os
import util # This needs to be imported before jax (at least as long as we are testing multi CPU/GPU devices)
import jax
from plotting import plot_history
import json_import
import time
import numpy as np
from time_message import tprint
# TODO rework import concept to be sth like from steps import NeuralField
from steps.GaussInput import GaussInput
from steps.NeuralField import NeuralField
from steps.StaticGain import StaticGain
from Architecture import Architecture
from AbsSigmoid import AbsSigmoid
from GaussKernel import GaussKernel

if __name__ == "__main__":

    ## --- Initialization ---

    use_cpu = False # Set this to true if you want jax to use CPU instead of GPU
    if use_cpu: # CPU
        jax.config.update('jax_platform_name', 'cpu')

    print("Computing devices found by JAX:")
    print(jax.local_devices())

    load_from_json = False

    if load_from_json: ## Load architecture from json file exported by CEDAR

        arch = json_import.import_file(os.path.join(util.root(), "cedar_architectures", "comp_tree.json"))
        step_to_plot = "Neural Field"

    else: ## Create architecture with code

        size = 50
        shape = (size, size)
        kernel_shape = (15, 15)
        sigma = 3
        kernel_amplitude = 0.018116
        amplitude = 1.0
        factor = 1.2

        # Example architecture 1
        if True:
            gi = GaussInput("gi", {"shape": shape, "sigma": sigma, "amplitude": amplitude})
            st = StaticGain("st", {"factor": factor})
            nf = NeuralField("nf", {"resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.1, 
                                    "input_noise_gain": 0.1, "sigmoid": AbsSigmoid(100, 0),
                                    "lateral_kernel_convolution": GaussKernel({"shape": kernel_shape,"sigma": sigma, "amplitude": kernel_amplitude}),
                                    "shape": shape})
            step_to_plot = "nf"

            arch = Architecture()
            arch.add_element(gi)
            arch.add_element(st)
            arch.add_element(nf)
            arch.connect_to("gi", "st")
            arch.connect_to("st", "nf")

        else: # Example architecture 2
            num_fields = 30

            arch = Architecture()
            gi = GaussInput("gi", {"shape": shape, "sigma": sigma, "amplitude": amplitude})
            arch.add_element(gi)
            for i in range(num_fields):
                nf = NeuralField(f"nf{i}", {"resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.1, 
                                    "input_noise_gain": 0.1, "sigmoid": AbsSigmoid(100, 0),
                                    "lateral_kernel_convolution": GaussKernel({"shape": kernel_shape,"sigma": sigma, "amplitude": kernel_amplitude}),
                                    "shape": shape})
                arch.add_element(nf)
                arch.connect_to("gi", f"nf{i}")
            step_to_plot = ""

    tprint("Architecture loaded")

    ## --- Compiling architecture ---

    arch.compile(warmup=True) # TODO this should be default ?
    tprint("Architecture compiled")

    ## --- Simulation ---

    num_steps = 10
    do_plot = True

    for i in range(2): # Do multiple runs to check stability of timing results
        print(f"\nRun {i+1}")
        history_new = arch.run_simulation(arch.tick, step_to_plot, num_steps)
        arch.reset_steps()

    print()
    tprint(f"Simulations done")

    ## --- Plotting ---
    if do_plot:
        plot_history(num_steps, history_new)
        tprint("Plot done")




# TODO
# Configurable
# tau ms?
# estimate width (GaussKernel) (and everything that is hardcoded in json_import (like kernel_amplitude))
# convolution?
# pmap / device_put?
# Some static steps (e.g. sources and those following) don't have to be computed every tick.
# Sources class without input_mat in compute_static?
# License

# Remember:
# In CEDAR if you create a Field and *then* change its resting_level, you might wanna hit reset before starting the simulation to fill the activation buffer with the desired resting_level values
# If you don't do it, the activation buffer needs many ticks to adapt to the new resting_level
# If you instead load the architecture from a file where a certain resting_level is already set for the field, the activation buffer will be initialized immediately (like in the reset case)
