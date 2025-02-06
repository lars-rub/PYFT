import argparse
import jax
from src.plotting import plot_history
from src.time_message import tprint
import time
import os
from src import util

if __name__ == "__main__":

    ## --- Argparse ---

    parser = argparse.ArgumentParser(description='CEDAR JAX')
    parser.add_argument('arch', type=str, help='Load architecture from JSON or Python file (i.e. JSON exported by CEDAR' + \
                        ' or python file containing get_architecture() function)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--plot_steps', default=[], nargs="+", help='Specify steps or their buffers to plot, e.g., \'step1 step2.buffer\' or \'"Neural Field1" "Static Gain1" "Neural Field2.activation"\' (without single quotes)')
    parser.add_argument('--run_for_n_ticks', type=int, default=10, help='Run simulation for n ticks')
    parser.add_argument('--save_plot', action='store_true', help="Save plots to 'output/plot_<timestamp>.png' instead of displaying them")
    parser.add_argument('--save_timing', action='store_true', help="Save timing results to 'output/timing_<timestamp>.txt'")
    parser.add_argument('--cache_jitted_funcs', action='store_true', help="Persistently save jit-compiled functions to reduce compilation time when running the same architecture multiple times")
    parser.add_argument('--arch_args', default=[], nargs="+", help='Optional arguments for the architecture that is loaded')
    parser.add_argument('--use_vmap', action='store_true', help="Use vmap instead of sequential computation")
    parser.add_argument('--eulerstep_static_args', action='store_true', help="Use static args in euler step. Does not work in combination with vmap")
    args = parser.parse_args()

    ## --- Initialization ---

    if args.use_vmap and args.eulerstep_static_args:
        raise Exception("vmap and eulerstep_statig_args cannot be used together")
    if args.cpu:
        jax.config.update('jax_platform_name', 'cpu')
    if args.cache_jitted_funcs:
        jax.config.update("jax_compilation_cache_dir", os.path.join(util.root(), "jax_cache"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print("Computing devices found by JAX:")
    print(jax.local_devices())
    if args.cpu:
        if not "Cpu" in str(jax.local_devices()):
            raise Exception("CPU not loaded. Make sure util_jax is not (in)directly imported before this line")

    # These imports have to happen *after* the jax config is set
    from src import util_jax
    import src.architecture_import as architecture_import

    util_jax.get_config()["use_vmap"] = args.use_vmap
    util_jax.get_config()["euler_step_partial"] = args.eulerstep_static_args

    ## --- Load architecture ---

    arch = architecture_import.import_file(args.arch, args.arch_args)
    tprint("Architecture loaded")

    ## --- Compile architecture ---

    compile_time = time.time()

    arch.compile()
    tprint("Architecture compiled")

    compile_time = time.time() - compile_time

    ## --- Simulation ---
    
    timings = []
    for i in range(2): # Do multiple runs to check stability of timing results
        print(f"\nRun {i+1}")
        plot_data_history, ms_per_tick = arch.run_simulation(arch.tick, args.plot_steps, args.run_for_n_ticks)
        arch.reset_steps()
        timings.append(ms_per_tick)

    print()
    tprint(f"Simulations done")

    ## --- Plotting ---
    if len(args.plot_steps) > 0:
        plot_history(args.run_for_n_ticks, plot_data_history, args.save_plot, args.plot_steps)
        tprint("Plot done")

    ## --- Save timing results ---

    if args.save_timing:
        with open(os.path.join(util.root(), "output", f"timing_{int(time.time())}.txt"), "w") as f:
            f.write(f"{sum(timings) / len(timings):>4.2f} ms per tick, {compile_time:>4.2f} s compilation, {os.path.basename(args.arch)}, {args.arch_args}, {'cpu' if args.cpu else 'GPU'}-{'vmap' if util_jax.cfg['use_vmap'] else 'static' if util_jax.cfg["euler_step_partial"] else 'sequential'}\n")




# TODO
# Configurable
# tau ms?
# convolution?
# pmap / device_put? --
# Some static steps (e.g. sources and those following) don't have to be computed every tick.
# Sources class without input_mat in compute_static?
# License

# Remember:
# In CEDAR if you create a Field and *then* change its resting_level, you might wanna hit reset before starting the simulation to fill the activation buffer with the desired resting_level values
# If you don't do it, the activation buffer needs many ticks to adapt to the new resting_level
# If you instead load the architecture from a file where a certain resting_level is already set for the field, the activation buffer will be initialized immediately (like in the reset case)
