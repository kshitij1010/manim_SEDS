import manim as mn
from gradient_decent import target_function, compute_gradient


class GradientDescentAnimation(mn.Scene):
    def construct(self):
        # Step 1: Set up axes and plot the target function
        axes = mn.Axes(
            x_range=[-5, 5, 1],  # Define x-axis range
            y_range=[0, 50, 10],  # Define y-axis range
            axis_config={"include_numbers": True}  # Show numbers on axes
        )
        graph = axes.plot(target_function, color=mn.BLUE)  # Plot the target function
        graph_label = axes.get_graph_label(graph, label="f(x)", x_val=-4, direction=mn.UP)
        self.add(axes, graph, graph_label)

        # Step 2: Initialize a starting point
        initial_x = -4.0
        tracker = mn.ValueTracker(initial_x)  # Track the current x value
        dot = mn.Dot(axes.c2p(initial_x, target_function(initial_x)), color=mn.RED)
        dot.add_updater(
            lambda d: d.move_to(axes.c2p(tracker.get_value(), target_function(tracker.get_value())))
        )
        self.add(dot)

        # Step 3: Gradient descent iterations
        learning_rate = 0.5
        num_steps = 10
        for _ in range(num_steps):
            current_x = tracker.get_value()
            grad = compute_gradient(current_x)  # Compute the gradient at the current x
            new_x = current_x - learning_rate * grad  # Update x using gradient descent

            # Step 4: Animate the update step
            arrow = mn.Arrow(
                start=axes.c2p(current_x, target_function(current_x)),
                end=axes.c2p(new_x, target_function(new_x)),
                buff=0,
                color=mn.YELLOW
            )
            self.play(mn.GrowArrow(arrow), run_time=0.5)
            self.play(tracker.animate.set_value(new_x), run_time=0.5)
            self.remove(arrow)

        # Step 5: Final point and label
        final_x = tracker.get_value()
        final_dot = mn.Dot(axes.c2p(final_x, target_function(final_x)), color=mn.GREEN)
        self.play(mn.FadeIn(final_dot))

        final_label = mn.Text(f"Minimum: ({final_x:.2f}, {target_function(final_x):.2f})").next_to(final_dot, mn.UP)
        self.play(mn.Write(final_label))