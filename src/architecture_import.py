from src.Architecture import Architecture
import json
import os

from src.steps.GaussInput import GaussInput
from src.steps.StaticGain import StaticGain
from src.steps.NeuralField import NeuralField
from src.AbsSigmoid import AbsSigmoid
from src.GaussKernel import GaussKernel

# TODO this creates lists for all keys, even if they are not duplicates. This is not a problem, but could be optimized
def _array_on_duplicate_keys(ordered_pairs):
    # Convert duplicate keys to arrays
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            d[k].append(v)
        else:
           d[k] = [v]
    return d

def _import_py_file(file_path, args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("arch", file_path)
    arch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch_module)
    get_architecture = arch_module.get_architecture
    arch = get_architecture(args)
    return arch

def _import_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file, object_pairs_hook=_array_on_duplicate_keys)
    steps= data["steps"][0]
    connections = data["connections"][0]
    arch = Architecture()

    for step_class in steps:
        for step_elem in steps[step_class]:
            if step_class == "cedar.processing.sources.GaussInput":
                gi = GaussInput(step_elem["name"][0], {"shape": [int(size) for size in step_elem["sizes"][0]], "sigma": float(step_elem["sigma"][0][0]), "amplitude": float(step_elem["amplitude"][0])})
                arch.add_element(gi)
            elif step_class == "cedar.processing.StaticGain":
                st = StaticGain(step_elem["name"][0], {"factor": float(step_elem["gain factor"][0])})
                arch.add_element(st)
            elif step_class == "cedar.dynamics.NeuralField":
                nf = NeuralField(step_elem["name"][0], {"resting_level": float(step_elem["resting level"][0]), 
                            "global_inhibition": float(step_elem["global inhibition"][0]), "tau": float(step_elem["time scale"][0]) / 1000, 
                            "input_noise_gain": float(step_elem["input noise gain"][0]), "sigmoid": AbsSigmoid(float(step_elem["sigmoid"][0]["beta"][0]), float(step_elem["sigmoid"][0]["threshold"][0])),
                            "lateral_kernel_convolution": GaussKernel({"sigma": float(step_elem["lateral kernels"][0]["cedar.aux.kernel.Gauss"][0]["sigmas"][0][0]), 
                            "amplitude": 0.018116}), "shape": [int(size) for size in step_elem["sizes"][0]]}) # float(step_elem["lateral kernels"][0]["cedar.aux.kernel.Gauss"][0]["amplitude"][0])
                # TODO dont hardcode 0.018116, instead get amplitude and normalization attribute
                arch.add_element(nf)
            else:
                raise Exception(f"Step {step_class} not known")

    for connection in connections:
        arch.connect_to(connection["source"][0].split(".")[0], connection["target"][0].split(".")[0])

    return arch

def import_file(file_path, args):
    ext = os.path.splitext(file_path)[1]
    if ext == ".json":
        return _import_json_file(file_path)
    elif ext == ".py":
        return _import_py_file(file_path, args)
    else:
        raise Exception(f"File extension {ext} not supported")
    