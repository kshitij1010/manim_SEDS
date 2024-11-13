from manim import *
import jax
import jax.numpy as jnp

def function(x):
    return jnp.sin(x)

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen