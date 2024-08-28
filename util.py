ROOT_FOLDER = ""
use_multiple_devices = False

import os
if not os.path.exists(os.path.join(ROOT_FOLDER, "run_config.json")):
    raise Exception(f"No run_config found in root folder {ROOT_FOLDER}. Please specify the correct path in the first line of util.py")

if use_multiple_devices:
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=3'
import json
import numpy as np
import jax.numpy as jnp
import jax

def _load_config():
    cfg = json.load(open(os.path.join(ROOT_FOLDER, "run_config.json"), "r"))

    # Parse dtype
    unsupported_dtypes = ["float64"]
    if "dtype" in cfg:
        dtype_str = cfg["dtype"]
        if cfg["dtype"] in unsupported_dtypes:
            raise Exception(f"dtype {cfg['dtype']} not supported")
        cfg["dtype"] = np.dtype(dtype_str).type
        cfg["jdtype"] = jnp.dtype(dtype_str).type
    return cfg

cfg = _load_config()

if cfg['debug']:
    jax_prng_key = jax.random.key(42)
    np.random.seed(42)
else:
    jax_prng_key = jax.random.key(np.random.randint(0, 2**16))

def root():
    return ROOT_FOLDER

def next_random_key():
    global jax_prng_key
    jax_prng_key, subkey = jax.random.split(jax_prng_key)
    return subkey

def next_random_keys(num):
    global jax_prng_key
    keys = jax.random.split(jax_prng_key, num+1)
    jax_prng_key = keys[0]
    return keys[1:]

# TODO the following function is very slow, not sure why though
# def next_random_keys_cached(num):
#     global jax_prng_key, pre_generated_keys
#     if pre_generated_keys is None or len(pre_generated_keys) < num:
#         pre_generated_keys = next_random_keys(num*10)
#         print("Generated new keys")
#     keys = pre_generated_keys[:num]
#     pre_generated_keys = pre_generated_keys[num:]
#     return keys

def prettify(l):
    return f'{np.mean(l):.5f} +- {np.std(l):.5f}  (min: {np.min(l)})'

def zeros(shape, device_idx=0):
    mat = jnp.zeros(shape, dtype=cfg["jdtype"])
    if device_idx != 0:
        mat = jax.device_put(mat, jax.local_devices()[device_idx])
    return mat

def ones(shape, device_idx=0):
    mat = jnp.ones(shape, dtype=cfg["jdtype"])
    if device_idx != 0:
        mat = jax.device_put(mat, jax.local_devices()[device_idx])
    return mat

gpu_idx = 0
def next_gpu():
    if not use_multiple_devices:
        return 0
    global gpu_idx
    gpu_idx += 1
    gpu_idx %= len(jax.local_devices())
    return gpu_idx


## --- Exceptions ---

class ArchitectureNotCompiledException(Exception):
    def __init__(self):
        super().__init__("Architecture needs to be compiled to perform this operation")


class ArchitectureCompiledException(Exception):
    def __init__(self):
        super().__init__("Architecture is already compiled, cannot perform this operation")
