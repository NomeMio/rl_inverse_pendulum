import json
import os

from cart_model import Cart_model
from continuous_policies import ActorPolicyContinuousSpace, ReinforcePolicy, SarsaTileCoding
from discrete_policies import ExpectedSarsaAgent, QLearningAgent, SarsaAgent
from train import evaluate_success_criterion, train_episodic_policy, train_step_policy

CONFIGS = {
    "sarsa": dict(
        cls=SarsaAgent, episodes=8000, driver=train_step_policy,
        kwargs=dict(gamma=0.98, alpha=0.1, epsilon_start=0.5, epsilon_min=0.05,
                    n_state_buckets=7, n_action_buckets=9),
    ),
    "q_learning": dict(
        cls=QLearningAgent, episodes=8000, driver=train_step_policy,
        kwargs=dict(gamma=0.98, alpha=0.1, epsilon_start=0.5, epsilon_min=0.05,
                    n_state_buckets=7, n_action_buckets=9),
    ),
    "expected_sarsa": dict(
        cls=ExpectedSarsaAgent, episodes=8000, driver=train_step_policy,
        kwargs=dict(gamma=0.98, alpha=0.1, epsilon_start=0.5, epsilon_min=0.05,
                    n_state_buckets=7, n_action_buckets=9),
    ),
    
    "reinforce": dict(
        cls=ReinforcePolicy, episodes=5000, driver=train_episodic_policy,
        kwargs=dict(alpha_rho=0.005, discount=0.99),
    ),
 
}


def main():
    context = Cart_model(human=False)
    results = {}
    for name, cfg in CONFIGS.items():
        print(f"=== {name} ===")
        policy = cfg["cls"](**cfg["kwargs"])
        cfg["driver"](policy, context, cfg["episodes"], max_steps=1000, desc=name)
        report = evaluate_success_criterion(policy, context, required_streak=100, max_steps=600, max_episodes=2000)
        policy.save(os.path.join("saved_policies", name))
        results[name] = report
        print(name, report)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    context.close()


if __name__ == "__main__":
    main()
