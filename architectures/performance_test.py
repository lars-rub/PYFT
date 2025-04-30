from src.steps.GaussInput import GaussInput
from src.steps.NeuralField import NeuralField
from src.steps.StaticGain import StaticGain
from src.Architecture import Architecture
from src.AbsSigmoid import AbsSigmoid
from src.GaussKernel import GaussKernel

# Supply args for this architecture using the --arch_args command line argument in the following format:
# num_fields shape kernel_sigma_shape gauss_mode
# e.g. '3 50x50 3x1 singlegauss' or '10 50x50x50 3x3x3 multigauss'
def get_architecture(args):
    if len(args) != 4:
        raise Exception("Expected exactly 4 arguments, got " + str(len(args)))
    
    # Parse arguments
    num_fields = int(args[0])
    shape = tuple([int(size) for size in args[1].split("x")])
    kernel_sigmas = tuple([float(size) for size in args[2].split("x")])
    gauss_input_sigma = tuple([float(size) for size in args[2].split("x")])
    if not args[3] in ["singlegauss", "multigauss"]:
        raise Exception("Expected 'singlegauss' or 'multigauss', got " + args[3])
    single_gauss = args[3] == "singlegauss"

    # Some static parameters
    kernel_amplitude = 1
    amplitude = 2
    factor = 2

    # Build architecture
    arch = Architecture()
    for i in range(num_fields):
        if i == 0 or not single_gauss:
            gi = GaussInput(f"gi{i}", {"shape": shape, "sigma": gauss_input_sigma, "amplitude": amplitude + i * 0.01})
            st = StaticGain(f"st{i}", {"factor": factor})
            arch.add_element(gi)
            arch.add_element(st)
            arch.connect_to(f"gi{i}", f"st{i}")
        nf = NeuralField(f"nf{i}", {"resting_level": -0.7+i*0.001, "global_inhibition": -0.01+i*0.001, "tau": 0.1, 
                            "input_noise_gain": 0.1+i*0.001, "sigmoid": AbsSigmoid(100+i*0.001, 0+i*0.001),
                            "lateral_kernel_convolution": 
                            GaussKernel({"sigma": kernel_sigmas, "amplitude": kernel_amplitude, "normalized": True, "max_shape": shape}),
                            "shape": shape})
        arch.add_element(nf)
        from_element = f"st0" if single_gauss else f"st{i}"
        arch.connect_to(from_element, f"nf{i}")
    return arch
