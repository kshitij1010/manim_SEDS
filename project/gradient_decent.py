from manim import *
import jax
import jax.numpy as jnp

# Define the target function
def target_function(x):
    """The target function we want to minimize."""
    return x**2 + 3 * x + 5

# Compute the gradient of the target function
def compute_gradient(x):
    """Computes the gradient of the target function at point x."""
    grad_fn = jax.grad(target_function)
    return grad_fn(x)

