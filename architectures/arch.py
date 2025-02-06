# TODO rework import concept to be sth like from steps import NeuralField
from src.steps.GaussInput import GaussInput
from src.steps.NeuralField import NeuralField
from src.steps.StaticGain import StaticGain
from src.Architecture import Architecture
from src.AbsSigmoid import AbsSigmoid
from src.GaussKernel import GaussKernel

def get_architecture(args):
    if len(args) != 3:
        raise Exception("Expected exactly 3 arguments, got " + str(len(args)))
    dimensionality = int(args[0])
    num_fields = int(args[1])
    size = int(args[2])
    shape = (size,) * dimensionality
    gauss_input_sigma = 3
    kernel_sigma = 3
    kernel_sigmas = (kernel_sigma,) * dimensionality
    kernel_amplitude = 0.018116
    amplitude = 2

    arch = Architecture()
    for i in range(num_fields):
        gi = GaussInput(f"gi{i}", {"shape": shape, "sigma": gauss_input_sigma, "amplitude": amplitude + i * 0.01})
        arch.add_element(gi)
        nf = NeuralField(f"nf{i}", {"resting_level": -0.7+i*0.001, "global_inhibition": -0.01+i*0.001, "tau": 0.1, 
                            "input_noise_gain": 0.1+i*0.001, "sigmoid": AbsSigmoid(100+i*0.001, 0+i*0.001),
                            "lateral_kernel_convolution": GaussKernel({"sigma": kernel_sigmas, "amplitude": kernel_amplitude}),
                            "shape": shape})
        arch.add_element(nf)
        arch.connect_to(f"gi{i}", f"nf{i}")
    return arch
