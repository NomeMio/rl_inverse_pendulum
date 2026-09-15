import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm

from train import evaluate_average_reward, train_episodic_policy, train_step_policy


def _train_policy(policy, context, episodes: int, max_steps: int, desc: str):
    """Dispatches to the per-step or episodic training loop based on
    whether the policy defines update_episode, so grid search keeps working
    if a policy's update style changes without anything here needing edits."""
    train_fn = train_episodic_policy if hasattr(policy, "update_episode") else train_step_policy
    train_fn(policy, context, episodes, max_steps=max_steps, desc=desc)


def _run_one_combo(policy_cls, params: dict, fixed_kwargs: dict, context_factory,
                    train_episodes: int, max_steps: int, eval_episodes: int, eval_max_steps: int,
                    desc: str):
    """One grid-search combo, self-contained so it can run in its own
    process: builds its own environment (a live gym/MuJoCo env can't be
    shared across processes), trains, evaluates, and closes it."""
    context = context_factory()
    try:
        policy = policy_cls(**{**fixed_kwargs, **params})
        _train_policy(policy, context, train_episodes, max_steps, desc=desc)
        metrics = evaluate_average_reward(policy, context, eval_episodes, max_steps=eval_max_steps)
    finally:
        context.close()
    return params, policy, metrics


def run_grid_search(policy_cls, context=None, param_grid: dict = None, fixed_kwargs: dict = None,
                     train_episodes: int = 1500, max_steps: int = 1000,
                     eval_episodes: int = 50, eval_max_steps: int = 1000,
                     final_train_episodes: int = None, desc: str = "grid_search",
                     parallel: bool = False, max_workers: int = None, context_factory=None):
    """
    Cartesian-product grid search over `param_grid` (hyperparameter name ->
    list of candidate values). For each combination: builds a fresh
    `policy_cls(**fixed_kwargs, **combo)`, trains it, and scores it with
    evaluate_average_reward over `eval_episodes` greedy episodes.

    Sequential mode (default): pass `context`, a single already-built
    environment reused for every combo, one after another.

    Parallel mode (`parallel=True`): combos train/evaluate concurrently in
    separate processes via ProcessPoolExecutor, each with its own
    environment -- a live gym/MuJoCo env instance isn't safe to share across
    processes. This requires `context_factory` instead of `context`: a
    zero-arg callable that builds a fresh context, e.g.
    `functools.partial(Cart_model, human=False)`. It must be importable
    (defined at module level, not a notebook-local lambda/closure), since
    worker processes need to pickle it -- same restriction applies to a
    custom `feature_fn` passed via `fixed_kwargs`, if the policy class takes
    one. Pick `max_workers` conservatively: MuJoCo stepping is CPU-bound, so
    more workers than physical cores won't help.

    If `final_train_episodes` is given, the winning combo is retrained from
    scratch at that larger episode budget (search-time budgets are meant to
    be small enough to compare many combos, not to fully train the winner)
    before being returned. This final retrain always runs in the calling
    process (not parallelized), using `context` if given, else a context
    built (and closed afterwards) from `context_factory`.

    Returns (best_params, best_policy, best_metrics, all_results), with
    all_results sorted by average_reward descending.
    """
    fixed_kwargs = fixed_kwargs or {}
    param_grid = param_grid or {}
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    combos = list(itertools.product(*value_lists))

    if parallel and context_factory is None:
        raise ValueError(
            "parallel=True requires context_factory (a picklable, importable zero-arg "
            "callable, e.g. functools.partial(Cart_model, human=False)) instead of "
            "context -- a live environment can't be shared across processes."
        )
    if not parallel and context is None:
        if context_factory is None:
            raise ValueError("run_grid_search needs either context or context_factory.")
        context = context_factory()

    results = []
    best_params = best_policy = best_metrics = None

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_combo, policy_cls, dict(zip(names, combo)), fixed_kwargs, context_factory,
                    train_episodes, max_steps, eval_episodes, eval_max_steps,
                    f"{desc} {idx + 1}/{len(combos)}",
                ): idx
                for idx, combo in enumerate(combos)
            }
            for future in as_completed(futures):
                idx = futures[future]
                params, policy, metrics = future.result()
                results.append({"params": params, **metrics})
                tqdm.write(f"[{desc}] {idx + 1}/{len(combos)} params={params} -> "
                           f"avg_reward={metrics['average_reward']:.2f} "
                           f"(min={metrics['min_reward']:.2f}, max={metrics['max_reward']:.2f})")
                if best_metrics is None or metrics["average_reward"] > best_metrics["average_reward"]:
                    best_params, best_policy, best_metrics = params, policy, metrics
    else:
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
        final_context = context if context is not None else context_factory()
        best_policy = policy_cls(**{**fixed_kwargs, **best_params})
        _train_policy(best_policy, final_context, final_train_episodes, max_steps, desc=f"{desc} retrain-best")
        best_metrics = evaluate_average_reward(best_policy, final_context, eval_episodes, max_steps=eval_max_steps)
        if final_context is not context:
            final_context.close()

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
    def grid_search(cls, context=None, param_grid: dict = None, fixed_kwargs: dict = None,
                     context_factory=None, parallel: bool = False, max_workers: int = None, **kwargs):
        return run_grid_search(cls, context=context, param_grid=param_grid or cls.default_param_grid(),
                                fixed_kwargs=fixed_kwargs, context_factory=context_factory,
                                parallel=parallel, max_workers=max_workers, **kwargs)
