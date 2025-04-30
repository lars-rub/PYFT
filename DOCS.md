# PYNFIT Documentation

This is a short in-progress documentation of PYNFIT.

## Using pynfit.py

Several arguments can be passed to pynfit.py to customize the simulation of architectures with PYNFIT.

### Mandatory Arguments

- `arch`\
The only mandatory argument to pynfit.py is the path to the architecture file.
This can be a JSON file (e.g. exported by CEDAR) or a python file that implements the get_architecture(args) method.
This method should create an Architecture and return it.
Examples for architecture files can be found in the `architectures` folder and a tutorial on creating architecture files can be found below.

### Optional Arguments

- `--cpu`\
Setting this flag forces JAX to use the CPU instead of the GPU.

- `--recording`\
The recording flag is used to specify which buffers of which steps should be recorded in each time step, to plot them later.
A list of names can be passed, with names being either step names, buffer names or slot names. Passing a step name (e.g. `field1`) will record the output matrix of the first output slot of that step. A slot name (e.g. `field1.output4`) will record the output of the specific slot of that step. A buffer name (e.g. `field1.activation`) will record the matrix of the given internal buffer of that step. Multiple names can be passed as a list (e.g. `--recording field1 static_gain2`).
Setting the `--recording` also automatically activates the plotting routine at the end of the simulation, resulting in the plot being displaying. If the plot should be saved instead, use the `--save_plot` flag.

- `--save_plot`\
Can be used in combination with `--recording` to save the plot instead of displaying it. The plot will be saved to `output/plot_<timestamp>.png`. The argument has no effect if used without `--recording`

- `--num_ticks`\
Sets the number of time steps the architecture should be simulated.

- `--num_runs`\
Sets the number of simulations that are done in total.
Doing multiple runs can be beneficial when doing time measurements.
All steps are reset after each run.

- `--cache_jitted_funcs`\
Setting this arguments results in JIT compiled functions being stored to disk. This reduces the compile time of subsequent runs of the same architecture. This will clutter up disk space if changes to JIT compilable functions are made often, but can make sense if large architectures are compiled multiple times without many changes.

- `--static_euler_compilation`\
With this argument, *all* euler computation functions of NeuralFields are compiled individually, requiring less variable arguments as its constant parameters can be fixed statically. Thus, a performance gain during the simulation can be achieved at the cost of increased compilation time. This can make sense for architectures that are already tuned and need to be simulated for a long time interval.

- `--arch_args`\
This can be used to supply arguments to a *.py* architecture file. A space-seperated list of strings can be provided (e.g. `--arch_args 3 5x5 1x1`) and a python list of all values (e.g. `["3", "5x5", "1x1"]`) will then be passed as the `args` argument to the `get_architecture()` function of the architecture.

### Tutorial: Creating a Python Architecture

PYNFIT can be used with either JSON or Python architectures.

While JSON files can, e.g., be exported using CEDAR, Python architectures allow the user to build a PYNFIT architecture using Python code.

An example showing the basic mechanics of creating a Python architecture can be found in `architectures/example_architecture.py`.

In this file we see that the get_architecture() method needs to be implemented.
This function retrieves optional architecture arguments from the command line as its parameter (see `--arch_args` above).

First, all Steps need to be instantiated and supplied with their respective parameters.

Each step needs to be added to the architecture by one of the four shown methods, each syntactically different but semantically the same.

Connecting the steps is also possible through various syntactic shortcuts, again with equal semantics.

The created architecture then needs to be returned to be later simulated by PYNFIT.

## For Developers

### JAX

All GPU accelerated computation in PYNFIT is based on JAX.

Reading the JAX documentation is recommended as it provides insights into the specifics of developing GPU optimized code with JAX. (https://docs.jax.dev/en/latest/, especially: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)

We will give a short overview of the most important aspects that are good to know when working with PYNFIT.

#### Handling Matrices

Matrix computation in JAX can be done using the jax.numpy (jnp) framework that uses similar syntax to numpy.

In many cases just swapping `np` by `jnp` is enough to enable matrix computation to be performed on the GPU.
Still, it is advised to read the related documentation (https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) as there are still some specifics that should be considered, depending on the use case at hand.

For example, printing the result of a jnp matrix after a computation can be done using `print(matrix.block_until_ready())`, which blocks the thread until the computation is ready, as most jnp functions do not block the thread but let the computation happen in the background to allow for automatic parallelization.

#### JIT Compilation

All functions that are to be JIT compiled by JAX (i.e., that should be run on the GPU) need the `@jax.jit` (or `@partial(jax.jit, ...)`) compiler directive. (JAX JIT docs: https://docs.jax.dev/en/latest/jit-compilation.html)

This mainly applies to the compute() function of steps as these are executed every tick of the simulation and usually contain the heavy parts of the computation.

The JIT compilation takes place when the function is called for the first time, which, in PYNFIT, is done during the warmup execution in the Architecture::compile() routine before the actual simulation starts, to not influence the timing measurements of the simulation.

During the JIT compilation, JAX builds a computation graph that describes how the output of the function (i.e., the return value) depends on the input (i.e., the arguments).
As only this direct correlation of input to output values is compiled and later used during the simulation, any side effects executed in the function (i.e., print() calls, setting global/member variables, ...) will not be executed during the simulation.

Thus, such programming concepts should be avoided and all JIT compilable functions should be functionally pure, i.e., all inputs that the function depends on should be passed as arguments while all outputs should be returned.
There are several other programming concepts that behave different than usual, e.g., if/else statements.
Please refer to (https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) for an overview on pure functions and some common gotchas of JAX that are important to consider.

Be careful when writing JIT compiled functions that are member of a class.

First, you should not access non-constant member variables of that class from within the function but instead pass them as input arguments if their value can change after function compilation.

Second, JAX can mainly handle numeric arguments or matrices, but cannot compile functions that take arguments like objects of a class.
Thus, e.g., the `self` argument, present in member functions, will throw an error when trying to JIT compile it.

This can be avoided by either making the function global and not part of the class (which may often be possible as it should only depend on its input arguments anyways), or by using the `@partial(jax.jit, static_argnames=['self'])` directive instead of `@jax.jit`.

This tells JAX to handle all arguments in the `static_argnames` list as static, and to recompile the function if it is called with a different value than before.
This means that for every value of `self` that the function will be called on, i.e., for each instance of the class, JAX saves a separate compiled function.

This can be problematic when used on arguments that are different every time the function is called, as it would recompiled each call, but can, however, be useful in cases where the value only takes on a small set of values (like class instances in the self variable) and is executed multiple times for each value.

### Class Overview

A short overview over the class structure of PYNFIT:

The Architecture class stores the architecture including all contained steps and connections.
Every Step has to be added to the architecture (Architecture::add_element()), and connections can be created using the Architecture::connect_to() method.

The Step class can be considered an abstract class that gives a basic interface for steps and provides functions like register_input/output or update_input.
Each individual step should be a class that inherits the Step class and implements the compute() method.

The Step class itself inherits the Configurable class, that handles all parameterizable objects (e.g., steps and GaussKernel).

The sigmoid used, e.g., by the NeuralField class, is currently handled by the class AbsSigmoid.
The actual sigmoid computation is, however, happening in sigmoids.py.
This might be reworked in the future.

### Tutorial: Creating a Step

Creating a new step involves creating a class that inherits the Step class and implements the compute() method.
Along with the current steps in use (e.g. GaussInput, NeuralField), two example steps can be found in `src/steps`, that are used in the `architectures/example_architecture.py`, along which we can learn hands-on how to create a static or dynamic step.

The first example is a static step (`ExampleStaticStep.py`) showing how to create a simple step with multiple inputs and outputs which performs basic matrix operations on the input matrices.
By inherting the Step class, the step automatically obtains a default input and output slot called `in0` and `out0` (see util::DEFAULT_INPUT_SLOT and util::DEFAULT_OUTPUT_SLOT).

The super constructor needs to be called with a list of mandatory parameters.
These can later be used without checking for their existence, as an error message would be thrown in the super constructor if they are not presetn in the params argument.
New inputs/outputs can be registered with the register_input()/register_output() methods.

A compute() method needs to be implemented.
This takes input_mats as an argument containing one input matrix for each input slot.
The `kwargs` argument is currently not used for static steps.
The `@partial(jax.jit, static_argnames=['self'])` directive allows this member function to be JIT compiled (see above)-
The two outputs are calculated based on matrix operations on the input matrices.
The first output uses simple linear combinations, while the second output shows the use of `jnp` to manipulate matrices.
Finally, output matrices for each output slot are returned.

The second example step is a dynamic step (`ExampleDynamicStep.py`).

It works similarly to the static step but needs the `is_dynamic=True` argument in the call to the super constructor to be dynamic.

Some other particularities of dynamic steps:
- Keyword arguments will be passed to the compute() function that might be required for dynamic computations.
Currently the only one being set (in the Architecture::tick() function) is `prng_key`, a JAX compatible random key.
- (optional) `self.needs_input_connections = False`\
This setting can be turned off for dynamic steps to allow this step to be used "standalone", not requiring inputs.
- (optional) `self._max_incoming_connections[util.DEFAULT_INPUT_SLOT] = jnp.inf`\
If multiple input connections for one input slot are allowed, this setting can be set to `jnp.inf`.
The individual inputs will automatically be summed up when retrieving them in the compute() function.

Other than that, the actual computation can be implemented in the compute() method similar to the static steps.
In the example steps, the random key is used to generate noise which is added to the input matrix.


### Things to Note

Currently, only one Architecture instance may be created during the runtime of the program.
If, in the future, there will be a requirement to instantiate multiple architectures, the util::set_architecture() mechanic should be reworked.
Currently, this mechanic is only used for the connection shortcuts (`step1 >> step2`) that replace the call `arch.connect_to("step1", "step2")`.
However, there may be different ways to implement this shortcut.
