import json
import os
import random

from cart_model import Action, State
from hyperparam_search import GridSearchMixin
from quantizer import ActionQuantizer, StateQuantizer


class TabularQPolicy(GridSearchMixin):
    """
    Shared base for tabular TD-control methods over a discretized state and
    action space. The state key uses all four state variables (including
    position: cart_model.py's Cart_model forcibly terminates an episode when
    the cart hits the position wall, so a policy blind to position cannot
    learn to avoid that failure mode). Subclasses only need to define the
    bootstrap target used in `update`.
    """

    @classmethod
    def default_param_grid(cls) -> dict:
        return {
            "gamma": [0.95, 0.99],
            "alpha": [0.05, 0.15],
            "n_state_buckets": [5, 9],
            "n_action_buckets": [7, 11],
        }

    def __init__(self, state_quantizer: StateQuantizer = None, action_quantizer: ActionQuantizer = None,
                 n_state_buckets: int = 7, n_action_buckets: int = 9,
                 gamma: float = 0.98, alpha: float = 0.1,
                 epsilon_start: float = 0.5, epsilon_min: float = 0.05, epsilon_decay_episodes: int = 3000,
                 alpha_min_fraction: float = 0.1, alpha_decay_episodes: int = 5000):
        self.state_quantizer = state_quantizer or StateQuantizer(n_buckets=n_state_buckets)
        self.action_quantizer = action_quantizer or ActionQuantizer(-3.0, 3.0, n_action_buckets)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.alpha_min_fraction = alpha_min_fraction
        self.alpha_decay_episodes = alpha_decay_episodes
        self.q_table = {}
        self.episodes_taken = 0

    def _state_key(self, state: State) -> tuple:
        return self.state_quantizer.discretize(state)

    def _q(self, key: tuple) -> list:
        return self.q_table.setdefault(key, [0.0] * self.action_quantizer.n_buckets)

    def current_epsilon(self) -> float:
        return max(self.epsilon_min, self.epsilon_start * (1 - self.episodes_taken / self.epsilon_decay_episodes))

    def current_alpha(self) -> float:
        return max(self.alpha_min_fraction * self.alpha, self.alpha / (1 + self.episodes_taken / self.alpha_decay_episodes))

    def new_episode(self, state: State = None):
        self.episodes_taken += 1

    def get_action(self, state: State, training: bool = False) -> Action:
        epsilon = self.current_epsilon() if training else 0.0
        q_values = self._q(self._state_key(state))
        if random.random() < epsilon:
            action_index = random.randrange(self.action_quantizer.n_buckets)
        else:
            max_q = max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == max_q]
            action_index = random.choice(best_indices)
        return Action(self.action_quantizer.bucket_center(action_index))

    def update(self, state: State, action: Action, reward: float, next_state: State, next_action: Action,
               terminated: bool = False):
        key = self._state_key(state)
        a_idx = self.action_quantizer.discretize(action.get_velocity())
        bootstrap = 0.0 if terminated else self.gamma * self._bootstrap_value(next_state, next_action)
        q = self._q(key)
        q[a_idx] += self.current_alpha() * (reward + bootstrap - q[a_idx])

    def _bootstrap_value(self, next_state: State, next_action: Action) -> float:
        raise NotImplementedError

    def save(self, dirname: str):
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "policy.json"), "w") as f:
            json.dump({
                "q_table": {str(k): v for k, v in self.q_table.items()},
                "gamma": self.gamma,
                "alpha": self.alpha,
                "epsilon_start": self.epsilon_start,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay_episodes": self.epsilon_decay_episodes,
                "alpha_min_fraction": self.alpha_min_fraction,
                "alpha_decay_episodes": self.alpha_decay_episodes,
            }, f)
        self.state_quantizer.save(os.path.join(dirname, "state_quantizer.json"))
        self.action_quantizer.save(os.path.join(dirname, "action_quantizer.json"))

    @classmethod
    def load(cls, dirname: str):
        import ast
        with open(os.path.join(dirname, "policy.json"), "r") as f:
            data = json.load(f)
        policy = cls.__new__(cls)
        policy.gamma = data["gamma"]
        policy.alpha = data["alpha"]
        policy.epsilon_start = data["epsilon_start"]
        policy.epsilon_min = data["epsilon_min"]
        policy.epsilon_decay_episodes = data["epsilon_decay_episodes"]
        policy.alpha_min_fraction = data["alpha_min_fraction"]
        policy.alpha_decay_episodes = data["alpha_decay_episodes"]
        policy.episodes_taken = 0
        policy.state_quantizer = StateQuantizer.load(os.path.join(dirname, "state_quantizer.json"))
        policy.action_quantizer = ActionQuantizer.load(os.path.join(dirname, "action_quantizer.json"))
        policy.q_table = {ast.literal_eval(k): v for k, v in data["q_table"].items()}
        return policy


class SarsaAgent(TabularQPolicy):
    def _bootstrap_value(self, next_state: State, next_action: Action) -> float:
        next_a_idx = self.action_quantizer.discretize(next_action.get_velocity())
        return self._q(self._state_key(next_state))[next_a_idx]


class QLearningAgent(TabularQPolicy):
    def _bootstrap_value(self, next_state: State, next_action: Action) -> float:
        return max(self._q(self._state_key(next_state)))


class ExpectedSarsaAgent(TabularQPolicy):
    def _bootstrap_value(self, next_state: State, next_action: Action) -> float:
        q = self._q(self._state_key(next_state))
        epsilon = self.current_epsilon()
        n = len(q)
        max_q = max(q)
        greedy_indices = [i for i, v in enumerate(q) if v == max_q]
        greedy_prob = (1 - epsilon) / len(greedy_indices)
        return sum((epsilon / n + (greedy_prob if i in greedy_indices else 0.0)) * v for i, v in enumerate(q))
