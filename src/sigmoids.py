import jax
import jax.numpy as jnp

@jax.jit
def AbsSigmoid(x, beta, theta):
    return 0.5 * (1.0 + beta * (x - theta) / (1.0 + beta * jnp.abs(x - theta)))

def AbsSigmoidWrapper(beta, theta):
    return lambda x: AbsSigmoid(x, beta, theta)