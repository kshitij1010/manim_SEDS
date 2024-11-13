from manim import *
import jax
import jax.numpy as jnp

STEP_SIZE = 0.1
MAX_ITER = 1000
SPEEDUP_STEP = 5
SEED = 0


def target_function(x):
    return jnp.sin(x)

class GradientDecent(Scene):
    def construct(self):

        # Plot the function

        # Initialize the starting point

        # Animate the gradient

        # Update the point

        # Repeat the process but speed up


        # delete the following line
        raise NotImplementedError("Please implement the code for plotting the function")
