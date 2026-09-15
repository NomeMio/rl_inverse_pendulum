"""
Visualization helpers for inspecting what a policy has actually learned,
without touching the gym/MuJoCo environment -- cart_model.State is built
directly from a plain observation list, so these can be called on a
policy at any point (mid-training, just-loaded, freshly initialized).
"""
import matplotlib.pyplot as plt
import numpy as np

from cart_model import State


def policy_action_heatmap(policy, angle_range=(-0.25, 0.25), angvel_range=(-3.0, 3.0),
                           n_angle=41, n_angvel=41, position=0.0, velocity=0.0,
                           ax=None, cmap="RdBu_r"):
    """
    Shows the action the policy would take (greedy, training=False) across a
    grid of (pole_angle, pole_angle_velocity), holding cart position/velocity
    fixed. Works for any policy exposing get_action(state, training) --
    discrete or continuous, tabular or linear -- since it only reads the
    resulting Action.get_velocity().

    A sane balancing policy should show a clear split: push right (positive,
    red) when the pole is leaning/falling right, push left (negative, blue)
    when it's leaning/falling left -- roughly antisymmetric through the
    origin. A policy that's actually state-independent (see the REINFORCE
    "shared" mode discussion) will show flat, angle/angvel-independent
    stripes instead.
    """
    angles = np.linspace(angle_range[0], angle_range[1], n_angle)
    angvels = np.linspace(angvel_range[0], angvel_range[1], n_angvel)
    grid = np.zeros((n_angvel, n_angle))
    for i, av in enumerate(angvels):
        for j, ang in enumerate(angles):
            state = State([position, ang, velocity, av])
            action = policy.get_action(state, training=False)
            grid[i, j] = action.get_velocity()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(grid.min()), abs(grid.max()), 1e-6)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                    extent=[angles[0], angles[-1], angvels[0], angvels[-1]])
    ax.set_xlabel("pole_angle (rad)")
    ax.set_ylabel("pole_angle_velocity (rad/s)")
    ax.set_title(f"{type(policy).__name__} chosen action (position={position}, velocity={velocity})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.colorbar(im, ax=ax, label="imprinted velocity (action)")
    return ax


def policy_action_distribution(policy, state, ax=None):
    """
    Bar chart of the full action-probability distribution at a single state,
    for policies with a discrete softmax action set (anything exposing
    _action_probs / _action_for_index, e.g. ReinforcePolicy in either
    weight_mode). Raises AttributeError for policies without a discrete
    distribution to show (e.g. a Gaussian continuous actor-critic) --
    use policy_action_heatmap for those instead.
    """
    if not (hasattr(policy, "_action_probs") and hasattr(policy, "_action_for_index")):
        raise AttributeError(
            f"{type(policy).__name__} has no discrete action distribution to plot "
            "(no _action_probs/_action_for_index) -- use policy_action_heatmap instead."
        )
    probs = policy._action_probs(state)
    centers = [policy._action_for_index(i).get_velocity() for i in range(len(probs))]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    width = (centers[1] - centers[0]) * 0.8 if len(centers) > 1 else 0.4
    ax.bar(centers, probs, width=width)
    ax.set_xlabel("action (imprinted velocity)")
    ax.set_ylabel("probability")
    ax.set_title(
        f"{type(policy).__name__} action distribution at "
        f"position={state.position:.2f}, velocity={state.velocity:.2f}, "
        f"pole_angle={state.pole_angle:.2f}, pole_angle_velocity={state.pole_angle_velocity:.2f}"
    )
    return ax
