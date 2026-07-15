import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm

import cart_model


def train_policy(policy, context: cart_model.Cart_model, episodes: int, desc: str = "train", disable: bool = False):
    for _ in tqdm(range(episodes), desc=desc, leave=False, disable=disable):
        policy.new_episode()
        state, _ = context.reset()
        action = policy.get_action(state, training=True)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            interaction = context.do_action(action)
            reward = interaction.reward
            next_state = interaction.state
            terminated = interaction.terminated
            truncated = interaction.truncated
            if terminated or truncated:
                policy.update(state, action, reward, next_state, None, True)
            else:
                next_action = policy.get_action(next_state, training=True)
                policy.update(state, action, reward, next_state, next_action, terminated)
                state = next_state
                action = next_action


def evaluate_policy(policy, context: cart_model.Cart_model, episodes: int, desc: str = "eval", disable: bool = False):
    rewards = []
    for _ in tqdm(range(episodes), desc=desc, leave=False, disable=disable):
        state, _ = context.reset()
        action = policy.egreedy_action(state, epsilon=0.0)
        total_reward = 0.0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            interaction = context.do_action(action)
            total_reward += interaction.reward
            terminated = interaction.terminated
            truncated = interaction.truncated
            if not (terminated or truncated):
                action = policy.egreedy_action(interaction.state, epsilon=0.0)
        rewards.append(total_reward)
    return {
        "average_reward": statistics.mean(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


def run_config(policy_class, params, extra_kwargs, context, train_episodes, eval_episodes,
                runs: int = 1, disable: bool = True, desc_prefix: str = ""):
    """
    Trains and evaluates `runs` independent policy instances for the same
    hyperparameters (fresh init + fresh training trajectory each time),
    and averages their eval stats. A single noisy train+eval run is a poor
    estimate of a hyperparameter setting's true performance (its score can
    swing a lot run to run); averaging several independent runs gives a
    much more reliable comparison between configs.

    Returns (aggregated_metrics, best_policy, per_run_metrics) where
    best_policy is the actual trained policy from the run with the highest
    average_reward among the `runs` repeats (there's no such thing as an
    "averaged" policy, so we keep the strongest concrete one).
    """
    run_metrics = []
    run_policies = []
    for r in range(runs):
        policy = policy_class(**params, **extra_kwargs)
        train_policy(policy, context, train_episodes, desc=f"{desc_prefix}run {r + 1}/{runs} train", disable=disable)
        metrics = evaluate_policy(policy, context, eval_episodes, desc=f"{desc_prefix}run {r + 1}/{runs} eval", disable=disable)
        run_metrics.append(metrics)
        run_policies.append(policy)

    aggregated = {
        "average_reward": statistics.mean(m["average_reward"] for m in run_metrics),
        "min_reward": statistics.mean(m["min_reward"] for m in run_metrics),
        "max_reward": statistics.mean(m["max_reward"] for m in run_metrics),
        "std_reward": statistics.pstdev(m["average_reward"] for m in run_metrics) if runs > 1 else 0.0,
    }
    best_index = max(range(runs), key=lambda i: run_metrics[i]["average_reward"])
    return aggregated, run_policies[best_index], run_metrics


def _run_trial(policy_class, params, extra_kwargs, train_episodes, eval_episodes, runs):
    """Runs `runs` independent train+eval repeats for one sampled config in
    its own process, with its own Cart_model (gym/mujoco environments
    aren't shareable across processes)."""
    context = cart_model.Cart_model(human=False)
    aggregated, best_policy, run_metrics = run_config(
        policy_class, params, extra_kwargs, context, train_episodes, eval_episodes, runs=runs, disable=True)
    context.close()
    return params, best_policy, aggregated, run_metrics


class RandomParameterSearch:
    """
    Random search over a policy class's hyperparameters.

    For each sampled parameter set, trains and evaluates `runs_per_trial`
    independent policy instances and averages their stats (see run_config),
    then keeps the best config by averaged reward. Ranges come from
    policy_class.get_paramters_samples(), a dict of name -> (low, high, step);
    values are snapped to the step grid, and cast to int when both low and
    step are int.
    """

    def __init__(self, policy_class, context: cart_model.Cart_model, param_ranges: dict = None,
                 n_trials: int = 20, train_episodes: int = 10000, eval_episodes: int = 200, seed: int = None,
                 extra_kwargs: dict = None, runs_per_trial: int = 1):
        self.policy_class = policy_class
        self.context = context
        self.param_ranges = param_ranges or policy_class.get_paramters_samples()
        self.n_trials = n_trials
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.rng = random.Random(seed)
        self.extra_kwargs = extra_kwargs or {}
        self.runs_per_trial = runs_per_trial
        self.results = []

    def _sample_params(self):
        params = {}
        for name, (low, high, step) in self.param_ranges.items():
            n_steps = round((high - low) / step)
            k = self.rng.randint(0, n_steps)
            value = low + k * step
            if isinstance(low, int) and isinstance(step, int):
                value = int(round(value))
            params[name] = value
        return params

    def run(self, verbose: bool = True, parallel: bool = False, n_workers: int = None):
        if parallel:
            return self._run_parallel(verbose=verbose, n_workers=n_workers)
        return self._run_sequential(verbose=verbose)

    def _run_sequential(self, verbose: bool = True):
        best_score = float("-inf")
        best_params = None
        best_policy = None
        best_metrics = None
        for trial in tqdm(range(self.n_trials), desc="param search"):
            params = self._sample_params()
            aggregated, policy, _ = run_config(
                self.policy_class, params, self.extra_kwargs, self.context,
                self.train_episodes, self.eval_episodes, runs=self.runs_per_trial,
                disable=False, desc_prefix=f"trial {trial + 1}/{self.n_trials} ")
            self.results.append({"params": params, **aggregated})
            if verbose:
                print(f"trial {trial + 1}/{self.n_trials}: params={params} -> "
                      f"avg_reward={aggregated['average_reward']:.2f} +/- {aggregated['std_reward']:.2f} "
                      f"(min={aggregated['min_reward']:.2f}, max={aggregated['max_reward']:.2f}, "
                      f"runs={self.runs_per_trial})")
            if aggregated["average_reward"] > best_score:
                best_score = aggregated["average_reward"]
                best_params = params
                best_policy = policy
                best_metrics = aggregated
        return best_params, best_policy, best_metrics, self.results

    def _run_parallel(self, verbose: bool = True, n_workers: int = None):
        n_workers = n_workers or os.cpu_count()
        trial_params = [self._sample_params() for _ in range(self.n_trials)]
        results = [None] * self.n_trials
        policies_by_trial = [None] * self.n_trials

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_trial, self.policy_class, params, self.extra_kwargs,
                                 self.train_episodes, self.eval_episodes, self.runs_per_trial): i
                for i, params in enumerate(trial_params)
            }
            for future in tqdm(as_completed(futures), total=self.n_trials, desc=f"param search (parallel x{n_workers})"):
                i = futures[future]
                params, policy, aggregated, _ = future.result()
                results[i] = {"params": params, **aggregated}
                policies_by_trial[i] = policy
                if verbose:
                    print(f"trial {i + 1}/{self.n_trials}: params={params} -> "
                          f"avg_reward={aggregated['average_reward']:.2f} +/- {aggregated['std_reward']:.2f} "
                          f"(min={aggregated['min_reward']:.2f}, max={aggregated['max_reward']:.2f}, "
                          f"runs={self.runs_per_trial})")

        self.results = results
        best_index = max(range(self.n_trials), key=lambda i: results[i]["average_reward"])
        best_entry = results[best_index]
        best_params = best_entry["params"]
        best_metrics = {k: v for k, v in best_entry.items() if k != "params"}
        best_policy = policies_by_trial[best_index]
        return best_params, best_policy, best_metrics, self.results
