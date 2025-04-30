from src.steps.GaussInput import GaussInput
from src.steps.NeuralField import NeuralField
from src.steps.StaticGain import StaticGain
from src.steps.ExampleStaticStep import ExampleStaticStep
from src.steps.ExampleDynamicStep import ExampleDynamicStep
from src.Architecture import Architecture
from src.AbsSigmoid import AbsSigmoid
from src.GaussKernel import GaussKernel

def get_architecture(args):
    arch = Architecture()
    shape = (50, 50)

    # Static steps
    gi0 = GaussInput("gi0", {"shape": shape, "sigma": (3,3), "amplitude": 1})
    gi1 = GaussInput("gi1", {"shape": shape, "sigma": (0.6,7), "amplitude": 2})
    st0 = StaticGain("st0", {"factor": 1.3})
    ss0 = ExampleStaticStep("ss0", {})

    # Add steps to architecture (different syntax possible)
    arch.add_element(gi0)
    arch = arch + gi1
    arch += st0
    arch + ss0

    # Dynamic steps
    ds0 = ExampleDynamicStep(f"ds0", {"shape": shape})
    nf0 = NeuralField(f"nf0", {"shape": shape, "resting_level": -0.7, "global_inhibition": -0.01, "tau": 0.1, 
                        "input_noise_gain": 0.1, "sigmoid": AbsSigmoid(100, 0),
                        "lateral_kernel_convolution": GaussKernel({"sigma": (3,3), "amplitude": 1, "normalized": True, "max_shape": shape}),})
    arch + ds0
    arch + nf0

    # Connections (different syntax possible)

    arch.connect_to("gi0", "st0")

    st0 >> ss0
    # => arch.connect_to("st0", "ss0")

    nf0.i0 << "ss0"
    # => arch.connect_to("ss0", "nf0")

    gi1 >> "ss0.second_input"
    # => arch.connect_to("gi1", "ss0.second_input")

    ds0 << ss0.o1
    # => arch.connect_to("ss0.second_output", "ds0")

    return arch
