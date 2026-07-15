import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import cart_model
import policies
import quantizer
import param_search

sns.set_theme(style="whitegrid")

STATE_NAMES = ["position", "velocity", "pole_angle", "pole_angle_velocity"]

N_TRIALS = 15
TRAIN_EPISODES = 100
RETRAIN_EPISODES = 20000
EVAL_EPISODES = 200
SAMPLE_EPISODES = 1000
STATE_BUCKETS = 10
RUNS_PER_TRIAL = 3
RETRAIN_RUNS = 3


def collect_samples(context, policy, episodes):
    state_samples = {name: [] for name in STATE_NAMES}
    action_samples = []
    for _ in range(episodes):
        episode = cart_model.Episode()
        while not episode.terminated and not episode.truncated:
            state, action, reward = episode.one_step(context, policy)
            state_samples["position"].append(state.position)
            state_samples["velocity"].append(state.velocity)
            state_samples["pole_angle"].append(state.pole_angle)
            state_samples["pole_angle_velocity"].append(state.pole_angle_velocity)
            action_samples.append(action.get_velocity())
    return state_samples, action_samples


def _bucket_counts_and_labels(bucketizer, values):
    boundaries = [bucketizer.low] + bucketizer.edges + [bucketizer.high]
    counts = [0] * len(bucketizer)
    for value in values:
        counts[bucketizer.index(value)] += 1
    labels = [f"[{boundaries[i]:.2f}, {boundaries[i + 1]:.2f})" for i in range(len(counts))]
    return counts, labels


def plot_state_distribution(filepath, title, state_quantizer, state_samples):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title)
    for ax, name in zip(axes.flat, STATE_NAMES):
        counts, labels = _bucket_counts_and_labels(state_quantizer.get_bucketizer(name), state_samples[name])
        sns.barplot(x=labels, y=counts, hue=labels, legend=False, palette="viridis", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("bucket")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_action_distribution(filepath, title, action_quantizer, action_samples):
    counts, labels = _bucket_counts_and_labels(action_quantizer.bucketizer, action_samples)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=labels, y=counts, hue=labels, legend=False, palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("action bucket (velocity)")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_param_search_results(filepath, results, title):
    trials = list(range(1, len(results) + 1))
    avg = [r["average_reward"] for r in results]
    mn = [r["min_reward"] for r in results]
    mx = [r["max_reward"] for r in results]
    param_names = list(results[0]["params"].keys())

    fig, axes = plt.subplots(1, 1 + len(param_names), figsize=(5 * (1 + len(param_names)), 5))

    ax0 = axes[0]
    ax0.errorbar(trials, avg, yerr=[[a - m for a, m in zip(avg, mn)], [x - a for a, x in zip(avg, mx)]],
                 fmt="none", ecolor="lightgray", capsize=4, zorder=1)
    sns.lineplot(x=trials, y=avg, marker="o", ax=ax0, color="tab:blue", zorder=2)
    ax0.set_xlabel("trial")
    ax0.set_ylabel("reward (avg, min-max range)")
    ax0.set_title("reward across trials")

    for ax, name in zip(axes[1:], param_names):
        values = [r["params"][name] for r in results]
        sns.scatterplot(x=values, y=avg, hue=avg, palette="viridis", legend=False, ax=ax)
        ax.set_xlabel(name)
        ax.set_ylabel("avg reward")
        ax.set_title(f"reward vs {name}")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_phase_comparison(filepath, phase_metrics: dict):
    names = list(phase_metrics.keys())
    avg = [phase_metrics[n]["average_reward"] for n in names]
    mn = [phase_metrics[n]["min_reward"] for n in names]
    mx = [phase_metrics[n]["max_reward"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=names, y=avg, hue=names, legend=False, palette="viridis", ax=ax)
    ax.errorbar(names, avg, yerr=[[a - m for a, m in zip(avg, mn)], [x - a for a, x in zip(avg, mx)]],
                 fmt="none", ecolor="black", capsize=5)
    ax.set_ylabel("average reward")
    ax.set_title("Best policy per run")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def run_phase(phase_name, run_dir, context, extra_kwargs=None):
    """Compares every action-bucket size from
    sarsaWithQuantization.get_action_bucket_sizes() under the same
    configuration: for each size, actions_buckets is held fixed and a full
    random search runs over the remaining hyperparameters (gamma, alpha).
    The bucket size with the best retrained metrics is kept as the phase's
    overall winner."""
    phase_dir = os.path.join(run_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    extra_kwargs = extra_kwargs or {}

    param_ranges = {name: bounds for name, bounds in policies.sarsaWithQuantization.get_paramters_samples().items()
                     if name != "actions_buckets"}
    bucket_sizes = policies.sarsaWithQuantization.get_action_bucket_sizes()

    bucket_comparison = {}
    overall = None  # (best_params, best_policy, best_metrics)

    for size in bucket_sizes:
        bucket_name = f"actions_buckets_{size}"
        bucket_dir = os.path.join(phase_dir, bucket_name)
        os.makedirs(bucket_dir, exist_ok=True)
        size_kwargs = {**extra_kwargs, "actions_buckets": size}

        searcher = param_search.RandomParameterSearch(
            policies.sarsaWithQuantization, context, param_ranges=param_ranges,
            n_trials=N_TRIALS, train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES,
            seed=0, extra_kwargs=size_kwargs, runs_per_trial=RUNS_PER_TRIAL,
        )
        best_params, _, search_best_metrics, results = searcher.run(verbose=True, parallel=False, n_workers=1)
        best_params = {**best_params, "actions_buckets": size}
        print(f"[{phase_name}][actions_buckets={size}] best params:", best_params,
              "search metrics:", search_best_metrics)

        # retrain the winning hyperparameters from scratch with a larger episode budget,
        # averaging RETRAIN_RUNS independent runs and keeping the strongest one
        best_metrics, best_policy, _ = param_search.run_config(
            policies.sarsaWithQuantization, best_params, extra_kwargs, context,
            RETRAIN_EPISODES, EVAL_EPISODES, runs=RETRAIN_RUNS, disable=False,
            desc_prefix=f"{phase_name} actions_buckets={size} retrain ",
        )
        print(f"[{phase_name}][actions_buckets={size}] retrained "
              f"({RETRAIN_EPISODES} episodes x{RETRAIN_RUNS} runs) metrics:", best_metrics)

        with open(os.path.join(bucket_dir, "param_search_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        plot_param_search_results(os.path.join(bucket_dir, "param_search_results.png"), results,
                                   f"{phase_name} (actions_buckets={size})")

        bucket_comparison[bucket_name] = best_metrics
        if overall is None or best_metrics["average_reward"] > overall[2]["average_reward"]:
            overall = (best_params, best_policy, best_metrics)

    best_params, best_policy, best_metrics = overall
    plot_phase_comparison(os.path.join(phase_dir, "actions_buckets_comparison.png"), bucket_comparison)
    print(f"[{phase_name}] best actions_buckets={best_params['actions_buckets']} "
          f"with params={best_params}, metrics={best_metrics}")

    random_state_samples, random_action_samples = collect_samples(context, None, SAMPLE_EPISODES)
    best_state_samples, best_action_samples = collect_samples(context, best_policy, SAMPLE_EPISODES)

    plot_state_distribution(os.path.join(phase_dir, "random_policy_state_distribution.png"),
                             "Random policy - state distribution", best_policy.quantizer, random_state_samples)
    plot_action_distribution(os.path.join(phase_dir, "random_policy_action_distribution.png"),
                              "Random policy - action distribution", best_policy.action_quantizer, random_action_samples)
    plot_state_distribution(os.path.join(phase_dir, "best_policy_state_distribution.png"),
                             "Best policy - state distribution", best_policy.quantizer, best_state_samples)
    plot_action_distribution(os.path.join(phase_dir, "best_policy_action_distribution.png"),
                              "Best policy - action distribution", best_policy.action_quantizer, best_action_samples)

    best_policy.save(os.path.join(phase_dir, "best_policy"))
    return best_params, best_policy, best_metrics, phase_dir


def focused_action_quantizer_factory(n_buckets):
    return quantizer.ActionQuantizer.focused_near_zero(-3.0, 3.0, n_buckets)


def main():
    run_dir = os.path.join("runs", f"sarsa_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    context = cart_model.Cart_model(human=False)
    phase_metrics = {}

    # --- Phase 3: linear (equal-width) quantization for state and action ---
    quantizer.StateQuantizer(n_buckets=STATE_BUCKETS).save("saved_policies/state_quantizer.json")
    if os.path.exists("saved_policies/action_quantizer.json"):
        os.remove("saved_policies/action_quantizer.json")

    linear_best_params, linear_best_policy, linear_metrics, linear_dir = run_phase(
        "linear_quantization", run_dir, context)
    phase_metrics["linear"] = linear_metrics

    # --- Phase 4: non-linear state quantization (quantile, fit on random rollouts, as in
    # test_bucket_distribution.py) + action quantization focused near 0 ---
    random_state_samples, _ = collect_samples(context, None, SAMPLE_EPISODES)
    quantile_state_quantizer = quantizer.StateQuantizer.from_samples(random_state_samples, n_buckets=STATE_BUCKETS)
    quantile_state_quantizer.save("saved_policies/state_quantizer.json")
    if os.path.exists("saved_policies/action_quantizer.json"):
        os.remove("saved_policies/action_quantizer.json")

    nonlinear_best_params, nonlinear_best_policy, nonlinear_metrics, nonlinear_dir = run_phase(
        "nonlinear_quantization", run_dir, context,
        extra_kwargs={"action_quantizer_factory": focused_action_quantizer_factory},
    )
    phase_metrics["nonlinear"] = nonlinear_metrics
    # persist the winning focused-near-zero action quantizer so plain construction
    # (no factory) reproduces it in phase 5
    nonlinear_best_policy.action_quantizer.save("saved_policies/action_quantizer.json")

    # --- Phase 5: rebalance the state quantization for the phase-4 winner, retrain from scratch ---
    rebalance_dir = os.path.join(run_dir, "nonlinear_rebalanced")
    os.makedirs(rebalance_dir, exist_ok=True)

    old_state_quantizer = nonlinear_best_policy.quantizer
    policy_state_samples, _ = collect_samples(context, nonlinear_best_policy, SAMPLE_EPISODES)

    plot_state_distribution(os.path.join(rebalance_dir, "old_state_distribution.png"),
                             "Old (phase 4) state quantization", old_state_quantizer, policy_state_samples)

    rebalanced_state_quantizer = quantizer.StateQuantizer.from_samples(
        policy_state_samples, n_buckets=old_state_quantizer.get_bucketizer("position").n_buckets)
    rebalanced_state_quantizer.save("saved_policies/state_quantizer.json")

    plot_state_distribution(os.path.join(rebalance_dir, "new_state_distribution.png"),
                             "New (rebalanced) state quantization", rebalanced_state_quantizer, policy_state_samples)

    metrics, retrained_policy, _ = param_search.run_config(
        policies.sarsaWithQuantization, nonlinear_best_params, {}, context,
        RETRAIN_EPISODES, EVAL_EPISODES, runs=RETRAIN_RUNS, disable=False,
        desc_prefix="nonlinear_rebalanced retrain ",
    )
    print(f"[nonlinear_rebalanced] retrained ({RETRAIN_EPISODES} episodes x{RETRAIN_RUNS} runs) metrics:", metrics)
    phase_metrics["nonlinear_rebalanced"] = metrics
    with open(os.path.join(rebalance_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    retrained_policy.save(os.path.join(rebalance_dir, "retrained_policy"))

    plot_phase_comparison(os.path.join(run_dir, "phase_comparison.png"), phase_metrics)

    print("run directory:", run_dir)


if __name__ == "__main__":
    main()
