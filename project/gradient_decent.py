from manim import *
import jax
import jax.numpy as jnp
import numpy as np

STEP_SIZE = 0.1
MAX_ITER = 10
SPEEDUP_STEP = 5
SEED = 129481

X_MIN = -3
X_MAX = 3

X_START = -2.7

def target_function(x):
    return 3 * x ** 2 + 2 * x + 1

target_grad = jax.grad(target_function)

class GradientDecent(Scene):
    def construct(self):

        # Plot the function

        ax = Axes(
            x_range=[X_MIN, X_MAX],
            y_range=[-10, 20],
            axis_config={"color": BLUE},
        )

        graph = ax.plot(target_function, color=WHITE)

        # Initialize the starting point

        initial_x = X_START
        initial_y = target_function(initial_x)
        point = Dot([ax.coords_to_point(initial_x, initial_y)])

        # Animate the gradient

        current_grad = target_grad(initial_x)
        unscaled_new_x = initial_x - current_grad
        new_x = initial_x - STEP_SIZE * current_grad
        unscaled_arrow = Arrow(
            start=point.get_center(),
            end=ax.coords_to_point(unscaled_new_x, target_function(unscaled_new_x)),
            color=RED,
        )
        arrow = Arrow(
            start=point.get_center(),
            end=ax.coords_to_point(new_x, target_function(new_x)),
            color=RED,
        )
        
        # Update the point

        t = ValueTracker(initial_x)
        point.add_updater(lambda m: m.move_to(ax.coords_to_point(t.get_value(), target_function(t.get_value()))))
        self.play(Create(ax))
        self.play(Create(graph))
        self.play(Create(point))
        self.play(Create(unscaled_arrow))
        self.play(Transform(unscaled_arrow,arrow))
        self.play(t.animate.set_value(new_x), run_time=1)

        # Repeat the process

        for i in range(MAX_ITER):
            current_grad = target_grad(t.get_value())
            unscaled_new_x = t.get_value() - current_grad
            new_x = t.get_value() - STEP_SIZE * current_grad
            unscaled_arrow.put_start_and_end_on(point.get_center(), ax.coords_to_point(new_x, target_function(new_x)))
            if i > SPEEDUP_STEP:
                self.play(t.animate.set_value(new_x), run_time=0.1)
            else:
                self.play(t.animate.set_value(new_x), run_time=1)
 