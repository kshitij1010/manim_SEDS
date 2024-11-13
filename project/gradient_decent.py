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
        x = ValueTracker(0)
        x.add_updater(lambda m, dt: m.increment_value(-STEP_SIZE * jax.grad(target_function)(m.get_value())))
        graph = self.get_graph(target_function, color=BLUE)
        dot = Dot().move_to(graph.points[0])
        self.play(Create(graph), Create(dot), run_time=0.1)
        self.play(x.set_value, 3, run_time=MAX_ITER * SPEEDUP_STEP)
        self.wait()

