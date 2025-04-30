import os
from src import util
import json
import numpy as np
import jax.numpy as jnp
import jax

def _load_config():
    cfg = json.load(open(os.path.join(util.root(), "run_config.json"), "r"))

    # Parse dtype
    unsupported_dtypes = ["float64"]
    if "dtype" in cfg:
        dtype_str = cfg["dtype"]
        if cfg["dtype"] in unsupported_dtypes:
            raise Exception(f"dtype {cfg['dtype']} not supported")
        cfg["dtype"] = np.dtype(dtype_str).type
        cfg["jdtype"] = jnp.dtype(dtype_str).type
    return cfg

_cfg_instance = None

def get_config():
    global _cfg_instance
    if _cfg_instance is None:
        _cfg_instance = _load_config()
    return _cfg_instance

cfg = get_config()

if cfg['debug']:
    jax_prng_key = jax.random.key(42)
    np.random.seed(42)
else:
    jax_prng_key = jax.random.key(np.random.randint(0, 2**16))

def next_random_key():
    global jax_prng_key
    jax_prng_key, subkey = jax.random.split(jax_prng_key)
    return subkey
def next_random_keys(num):
    global jax_prng_key
    keys = jax.random.split(jax_prng_key, num+1)
    jax_prng_key = keys[0]
    return keys[1:]

def zeros(shape):
    mat = jnp.zeros(shape, dtype=cfg["jdtype"])
    return mat

def ones(shape):
    mat = jnp.ones(shape, dtype=cfg["jdtype"])
    return mat
