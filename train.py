import statistics

from tqdm.auto import tqdm, trange

from cart_model import Cart_model


def _progress_interval(episodes: int, print_every: int = None, target_prints: int = 20) -> int:
    if print_every is not None:
        return max(1, print_every)
    return max(1, episodes // target_prints)


def train_step_policy(policy, context: Cart_model, episodes: int, max_steps: int = 1000, desc: str = "train",
                       print_every: int = None):
    """Per-step training loop for SARSA, Q-learning, Expected SARSA,
    Actor-Critic and tile-coded SARSA (any policy whose update() takes
    (state, action, reward, next_state, next_action, terminated))."""
    interval = _progress_interval(episodes, print_every)
    recent_lengths = []
    bar = trange(episodes, desc=desc, leave=False)
    for ep in bar:
        policy.new_episode()
        state, _ = context.reset()
        action = policy.get_action(state, training=True)
        survived = 0
        for _ in range(max_steps):
            interaction = context.do_action(action)
            survived += 1
            reward = interaction.reward
            next_state = interaction.state
            if interaction.terminated or interaction.truncated:
                policy.update(state, action, reward, next_state, None, interaction.terminated)
                break
            next_action = policy.get_action(next_state, training=True)
            policy.update(state, action, reward, next_state, next_action, interaction.terminated)
            state, action = next_state, next_action
        recent_lengths.append(survived)
        if (ep + 1) % interval == 0 or ep + 1 == episodes:
            avg_len = statistics.mean(recent_lengths)
            tqdm.write(f"[{desc}] episode {ep + 1}/{episodes} ({100 * (ep + 1) / episodes:.0f}%) "
                       f"avg_steps_survived={avg_len:.1f}")
            recent_lengths = []


def train_episodic_policy(policy, context: Cart_model, episodes: int, max_steps: int = 1000, desc: str = "train",
                           print_every: int = None):
    """Episodic (Monte Carlo) training loop for REINFORCE: collects a full
    trajectory and its rewards, then calls policy.update_episode(...)."""
    interval = _progress_interval(episodes, print_every)
    recent_lengths = []
    recent_rewards = []
    bar = trange(episodes, desc=desc, leave=False)
    for ep in bar:
        policy.new_episode()
        state, _ = context.reset()
        trajectory = []
        rewards = []
        for _ in range(max_steps):
            action = policy.get_action(state, training=True)
            trajectory.append((state, action))
            interaction = context.do_action(action)
            rewards.append(interaction.reward)
            state = interaction.state
            if interaction.terminated or interaction.truncated:
                break
        policy.update_episode(trajectory, rewards)
        recent_lengths.append(len(trajectory))
        recent_rewards.append(sum(rewards))
        if (ep + 1) % interval == 0 or ep + 1 == episodes:
            avg_len = statistics.mean(recent_lengths)
            avg_reward = statistics.mean(recent_rewards)
            tqdm.write(f"[{desc}] episode {ep + 1}/{episodes} ({100 * (ep + 1) / episodes:.0f}%) "
                       f"avg_steps_survived={avg_len:.1f} avg_reward={avg_reward:.1f}")
            recent_lengths = []
            recent_rewards = []


def evaluate_average_reward(policy, context: Cart_model, episodes: int, max_steps: int = 1000) -> dict:
    """Quick dev-time sanity metric: mean/min/max total reward over greedy
    (training=False) episodes."""
    rewards = []
    for _ in trange(episodes, desc="eval", leave=False):
        state, _ = context.reset()
        action = policy.get_action(state, training=False)
        total_reward = 0.0
        for _ in range(max_steps):
            interaction = context.do_action(action)
            total_reward += interaction.reward
            if interaction.terminated or interaction.truncated:
                break
            action = policy.get_action(interaction.state, training=False)
        rewards.append(total_reward)
    return {
        "average_reward": statistics.mean(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


def evaluate_success_criterion(policy, context: Cart_model, required_streak: int = 100, max_steps: int = 600,
                                max_episodes: int = 2000, print_every: int = None) -> dict:
    """Runs greedy episodes until `required_streak` consecutive episodes each
    survive `max_steps` ticks without terminating/truncating (this includes
    cart_model.py's position-wall SpaceLimitTruncation, since that sets
    interaction.terminated/truncated like any other end-of-episode signal),
    or `max_episodes` is exhausted."""
    interval = _progress_interval(max_episodes, print_every)
    streak = 0
    longest_streak = 0
    survival_lengths = []
    episodes_run = 0
    bar = trange(max_episodes, desc="success check", leave=False)
    for ep in bar:
        episodes_run += 1
        state, _ = context.reset()
        action = policy.get_action(state, training=False)
        survived = 0
        ended_early = False
        for step in range(max_steps):
            interaction = context.do_action(action)
            survived += 1
            if interaction.terminated or interaction.truncated:
                ended_early = True
                break
            action = policy.get_action(interaction.state, training=False)
        survival_lengths.append(survived)
        if ended_early:
            streak = 0
        else:
            streak += 1
            longest_streak = max(longest_streak, streak)
        if (ep + 1) % interval == 0 or streak >= required_streak or ep + 1 == max_episodes:
            tqdm.write(f"[success check] episode {ep + 1}/{max_episodes} current_streak={streak} "
                       f"longest_streak={longest_streak} last_survival={survived}")
        if streak >= required_streak:
            break
    return {
        "met": streak >= required_streak,
        "episodes_run": episodes_run,
        "longest_streak": longest_streak,
        "mean_survival": statistics.mean(survival_lengths),
    }
