# PYNFIT (PYamic Field Theory)

Welcome to PYNFIT, the GPU accelerated CEDAR prototype!

## Requirements

- Insert your root path (folder that contains run_config.json) at the top of util.py, or leave it empty if you plan to run the main script from the root path
- Install the required pip packages by running `pip install -r requirements.txt`
- Run pynfit.py (see Usage)

## Usage

PYNFIT requires an architecture file (JSON or .py file) as a mandatory argument and will simulate the architecture on the GPU using JAX.

Several optional arguments exist to adjust parameters of the simulation.
While we only give a short overview on the usage of the most important arguments here, there is a detailed overview of all parameters in *DOCS.md*.

The number of ticks the architecture is run can be set with the parameter `--num_ticks`.

To plot the output of steps (or internal buffers) a list of names of the steps/slots/buffers can be passed to `--recording`, e.g., `--recording static_gain0 field2 field2.activation`.

When the `--recording` flag is used a plot will show after the simulation displaying the corresponding matrices.
If instead the plot should be saved, the `--save_plot` trigger can be set.
Be careful with plots of matrices with a dimensionality of more than 3 as these get reduced to 3D plots to be able to plot them.
The reduction performed may not give a meaningful representation of the actual matrix.

If you want to pass arguments to your python architecture file you can use the `--arch_args` argument.

The `--cache_jitted_funcs` argument can be passed to allow JIT compiled functions to be saved to disk, reducing compile time in subsequent runs of, but requiring disk space. This makes sense if large architectures with long compile times are compiled multiple times (i.e., pynfit.py is executed multiple times) without big changes to the architecture.

When the architecture is tuned and should now be run for a large number of iterations, the `--static_euler_compilation` argument can be set. This often improves runtime performance of the simulation but may increase compilation time, as it pre-compiles *every* individual euler function of neural fields so that their parameters can stay constant.


## Developing

A detailed explanation of all command line arguments can be found in *DOCS.md*.

Tutorials on how to create architectures and new steps can be found there as well.
