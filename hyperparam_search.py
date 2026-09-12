import itertools

from tqdm.auto import tqdm

from train import evaluate_average_reward, train_episodic_policy, train_step_policy


def _train_policy(policy, context, episodes: int, max_steps: int, desc: str):
    """Dispatches to the per-step or episodic training loop based on
    whether the policy defines update_episode, so grid search keeps working
    if a policy's update style changes without anything here needing edits."""
    train_fn = train_episodic_policy if hasattr(policy, "update_episode") else train_step_policy
    train_fn(policy, context, episodes, max_steps=max_steps, desc=desc)


def run_grid_search(policy_cls, context, param_grid: dict, fixed_kwargs: dict = None,
                     train_episodes: int = 1500, max_steps: int = 1000,
                     eval_episodes: int = 50, eval_max_steps: int = 1000,
                     final_train_episodes: int = None, desc: str = "grid_search"):
    """
    Cartesian-product grid search over `param_grid` (hyperparameter name ->
    list of candidate values). For each combination: builds a fresh
    `policy_cls(**fixed_kwargs, **combo)`, trains it, and scores it with
    evaluate_average_reward over `eval_episodes` greedy episodes. Prints one
    progress line per combination.

    If `final_train_episodes` is given, the winning combo is retrained from
    scratch at that larger episode budget (search-time budgets are meant to
    be small enough to compare many combos, not to fully train the winner)
    before being returned.

    Returns (best_params, best_policy, best_metrics, all_results), with
    all_results sorted by average_reward descending.
    """
    fixed_kwargs = fixed_kwargs or {}
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    combos = list(itertools.product(*value_lists))

    results = []
    best_params = best_policy = best_metrics = None

    for idx, combo in enumerate(combos):
        params = dict(zip(names, combo))
        policy = policy_cls(**{**fixed_kwargs, **params})
        _train_policy(policy, context, train_episodes, max_steps, desc=f"{desc} {idx + 1}/{len(combos)}")
        metrics = evaluate_average_reward(policy, context, eval_episodes, max_steps=eval_max_steps)
        results.append({"params": params, **metrics})
        tqdm.write(f"[{desc}] {idx + 1}/{len(combos)} params={params} -> "
                   f"avg_reward={metrics['average_reward']:.2f} "
                   f"(min={metrics['min_reward']:.2f}, max={metrics['max_reward']:.2f})")
        if best_metrics is None or metrics["average_reward"] > best_metrics["average_reward"]:
            best_params, best_policy, best_metrics = params, policy, metrics

    results.sort(key=lambda r: r["average_reward"], reverse=True)

    if final_train_episodes:
        best_policy = policy_cls(**{**fixed_kwargs, **best_params})
        _train_policy(best_policy, context, final_train_episodes, max_steps, desc=f"{desc} retrain-best")
        best_metrics = evaluate_average_reward(best_policy, context, eval_episodes, max_steps=eval_max_steps)

    return best_params, best_policy, best_metrics, results


class GridSearchMixin:
    """
    Mixin that gives a policy class a `grid_search` classmethod for free, as
    long as it implements `default_param_grid()`. Deliberately thin: it just
    forwards to run_grid_search(), which only assumes the class is
    constructible from keyword arguments -- so this mixin, and grid search
    itself, keep working across future changes to a policy's constructor,
    hyperparameters, or update style without needing to be touched.
    """

    @classmethod
    def default_param_grid(cls) -> dict:
        raise NotImplementedError

    @classmethod
    def grid_search(cls, context, param_grid: dict = None, fixed_kwargs: dict = None, **kwargs):
        return run_grid_search(cls, context, param_grid or cls.default_param_grid(),
                                fixed_kwargs=fixed_kwargs, **kwargs)
