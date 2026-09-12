import json
import random
from math import exp

from cart_model import Action, State
from hyperparam_search import GridSearchMixin
from quantizer import ActionQuantizer
from tile_coding import TileCoder


class ActorPolicyContinuousSpace(GridSearchMixin):
    """
    One-step actor-critic with a Gaussian policy over linear features.

    `feature_fn(state) -> list[float]` produces the full raw feature vector
    (defaults to [position, velocity, pole_angle, pole_angle_velocity]).
    Which of *those* features feed the value function (`w`), the policy mean
    (`rho_m`) and the policy log-std (`rho_s`, exponentiated) is chosen via
    index lists (`value_features`, `mean_features`, `std_features`) into
    whatever `feature_fn` returns — so growing/shrinking/reordering the
    feature set only means passing a different `feature_fn` (and matching
    index lists); nothing else in the class assumes a fixed count of 4.
    Defaults reproduce the original fixed split: value_features=[0,1,2,3],
    mean_features=[0,2] (position, pole_angle), std_features=[1,3]
    (velocity, pole_angle_velocity).
    """

    DEFAULT_FEATURE_NAMES = ["position", "velocity", "pole_angle", "pole_angle_velocity"]

    @staticmethod
    def default_feature_fn(state: State) -> list:
        return [state.position, state.velocity, state.pole_angle, state.pole_angle_velocity]

    @classmethod
    def default_param_grid(cls) -> dict:
        return {
            "alpha_w": [0.02, 0.05, 0.1],
            "alpha_rho": [0.005, 0.01, 0.02],
            "discount": [0.95, 0.99],
        }

    def __init__(self, alpha_w: float = 0.1, alpha_rho: float = 0.1, discount: float = 0.85,
                 value_features: list = None, mean_features: list = None, std_features: list = None,
                 feature_fn=None):
        self.feature_fn = feature_fn or self.default_feature_fn
        self.value_features = list(value_features) if value_features is not None else [0, 1, 2, 3]
        self.mean_features = list(mean_features) if mean_features is not None else [0, 2]
        self.std_features = list(std_features) if std_features is not None else [1, 3]

        self.rho_m_size = len(self.mean_features)
        self.rho_s_size = len(self.std_features)
        self.rho_size = self.rho_m_size + self.rho_s_size
        self.rho = [random.uniform(-1, 1) for _ in range(self.rho_size)]
        self.alpha_w = alpha_w
        self.i = 1.0
        self.alpha_rho = alpha_rho
        self.discount = discount
        self.w = [random.uniform(-1, 1) for _ in range(len(self.value_features))]

    def _raw_features(self, state: State) -> list:
        return self.feature_fn(state)

    def value(self, state: State) -> float:
        features = self.get_features(state)
        return sum(self.w[i] * features[i] for i in range(len(features)))

    def m(self, state: State) -> float:
        rho_m = self.get_rho_m()
        features_rho_m = self.get_features_rho_m(state)
        return sum(rho_m[i] * features_rho_m[i] for i in range(len(rho_m)))

    def s(self, state: State) -> float:
        rho_s = self.get_rho_s()
        features_rho_s = self.get_features_rho_s(state)
        s = sum(rho_s[i] * features_rho_s[i] for i in range(len(rho_s)))
        s = max(-1.0, min(2.0, s))
        return exp(s)

    def new_episode(self, state: State = None):
        self.i = 1.0

    def get_action(self, state: State, training: bool = False) -> Action:
        action_value = random.gauss(self.m(state), self.s(state)) if training else self.m(state)
        return Action(action_value)

    def update(self, state: State, action: Action, reward: float, next_state: State, next_action: Action,
               terminated: bool = False):
        bootstrap = 0.0 if terminated else self.value(next_state)
        delta = reward + self.discount * bootstrap - self.value(state)
        features = self.get_features(state)
        for i in range(len(self.w)):
            self.w[i] += self.alpha_w * delta * features[i]
        a = action.get_velocity()
        m_s, s_s = self.m(state), self.s(state)
        rho_m_factor = (1 / s_s ** 2) * (a - m_s)
        rho_s_factor = ((a - m_s) ** 2 / s_s ** 2) - 1
        for i in range(self.rho_m_size):
            self.rho[i] += self.alpha_rho * delta * self.i * rho_m_factor * self.get_features_rho_m(state)[i]
        for i in range(self.rho_s_size):
            self.rho[i + self.rho_m_size] += self.alpha_rho * delta * self.i * rho_s_factor * self.get_features_rho_s(state)[i]
        self.i *= self.discount

    def get_features(self, state: State) -> list:
        raw = self._raw_features(state)
        return [raw[i] for i in self.value_features]

    def get_features_rho_m(self, state: State) -> list:
        raw = self._raw_features(state)
        return [raw[i] for i in self.mean_features]

    def get_features_rho_s(self, state: State) -> list:
        raw = self._raw_features(state)
        return [raw[i] for i in self.std_features]

    def get_rho_m(self) -> list:
        return self.rho[:self.rho_m_size]

    def get_rho_s(self) -> list:
        return self.rho[self.rho_m_size:self.rho_m_size + self.rho_s_size]

    def save(self, dirname: str):
        import os
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "policy.json"), "w") as f:
            json.dump({
                "w": self.w,
                "rho": self.rho,
                "alpha_w": self.alpha_w,
                "alpha_rho": self.alpha_rho,
                "discount": self.discount,
                "value_features": self.value_features,
                "mean_features": self.mean_features,
                "std_features": self.std_features,
            }, f)

    @classmethod
    def load(cls, dirname: str, feature_fn=None):
        """`feature_fn` must be passed again if a non-default one was used to
        train the saved policy — it's a callable, so it isn't persisted in
        the JSON file."""
        import os
        with open(os.path.join(dirname, "policy.json"), "r") as f:
            data = json.load(f)
        policy = cls(alpha_w=data["alpha_w"], alpha_rho=data["alpha_rho"], discount=data["discount"],
                     value_features=data["value_features"], mean_features=data["mean_features"],
                     std_features=data["std_features"], feature_fn=feature_fn)
        policy.w = data["w"]
        policy.rho = data["rho"]
        return policy


class ReinforcePolicy(GridSearchMixin):
    """
    Episodic (Monte Carlo) REINFORCE with a Gaussian policy over linear
    features, no value baseline. Same mean/log-std parameterization as
    ActorPolicyContinuousSpace, but the update happens once per episode
    using the full discounted returns instead of a TD bootstrap.
    """

    @classmethod
    def default_param_grid(cls) -> dict:
        return {
            "alpha_rho": [0.001, 0.005, 0.01],
            "discount": [0.95, 0.99],
        }

    def __init__(self, alpha_rho: float = 0.005, discount: float = 0.99):
        self.rho_m_size = 2
        self.rho_s_size = 2
        self.rho_size = 4
        self.rho = [random.uniform(-1, 1) for _ in range(self.rho_size)]
        self.alpha_rho = alpha_rho
        self.discount = discount

    def m(self, state: State) -> float:
        rho_m = self.get_rho_m()
        features_rho_m = self.get_features_rho_m(state)
        return sum(rho_m[i] * features_rho_m[i] for i in range(len(rho_m)))

    def s(self, state: State) -> float:
        rho_s = self.get_rho_s()
        features_rho_s = self.get_features_rho_s(state)
        s = sum(rho_s[i] * features_rho_s[i] for i in range(len(rho_s)))
        s = max(-1.0, min(2.0, s))
        return exp(s)

    def new_episode(self, state: State = None):
        pass

    def get_action(self, state: State, training: bool = False) -> Action:
        action_value = random.gauss(self.m(state), self.s(state)) if training else self.m(state)
        return Action(action_value)

    def update_episode(self, trajectory: list, rewards: list):
        n = len(trajectory)
        returns = [0.0] * n
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = rewards[t] + self.discount * running
            returns[t] = running
        discount_pow = 1.0
        for t in range(n):
            state, action = trajectory[t]
            a = action.get_velocity()
            m_s, s_s = self.m(state), self.s(state)
            grad_m = (a - m_s) / s_s ** 2
            grad_s = ((a - m_s) ** 2 / s_s ** 2) - 1
            coeff = self.alpha_rho * discount_pow * returns[t]
            for i in range(self.rho_m_size):
                self.rho[i] += coeff * grad_m * self.get_features_rho_m(state)[i]
            for i in range(self.rho_s_size):
                self.rho[i + self.rho_m_size] += coeff * grad_s * self.get_features_rho_s(state)[i]
            discount_pow *= self.discount

    def get_features_rho_m(self, state: State) -> list:
        return [state.position, state.pole_angle]

    def get_features_rho_s(self, state: State) -> list:
        return [state.velocity, state.pole_angle_velocity]

    def get_rho_m(self) -> list:
        return [self.rho[0], self.rho[1]]

    def get_rho_s(self) -> list:
        return [self.rho[2], self.rho[3]]

    def save(self, dirname: str):
        import os
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "policy.json"), "w") as f:
            json.dump({"rho": self.rho, "alpha_rho": self.alpha_rho, "discount": self.discount}, f)

    @classmethod
    def load(cls, dirname: str):
        import os
        with open(os.path.join(dirname, "policy.json"), "r") as f:
            data = json.load(f)
        policy = cls(alpha_rho=data["alpha_rho"], discount=data["discount"])
        policy.rho = data["rho"]
        return policy


class SarsaTileCoding(GridSearchMixin):
    """
    Semi-gradient SARSA over a tile-coded representation of the continuous
    4D state, with a discretized action space (reusing ActionQuantizer).
    On-policy TD control: Q(s,a) = sum(w[a][i] for i in active tile features).
    """

    STATE_RANGES = [(-0.4, 0.4), (-4.0, 4.0), (-0.5, 0.5), (-7.0, 7.0)]

    @classmethod
    def default_param_grid(cls) -> dict:
        return {
            "alpha": [0.05, 0.1, 0.2],
            "gamma": [0.95, 0.99],
            "n_tilings": [4, 8],
            "tiles_per_dim": [4, 6],
        }

    def __init__(self, action_quantizer: ActionQuantizer = None, n_action_buckets: int = 9,
                 n_tilings: int = 8, tiles_per_dim: int = 6,
                 gamma: float = 0.98, alpha: float = 0.1,
                 epsilon_start: float = 0.3, epsilon_min: float = 0.02, epsilon_decay_episodes: int = 2000):
        self.action_quantizer = action_quantizer or ActionQuantizer(-3.0, 3.0, n_action_buckets)
        self.tile_coder = TileCoder(self.STATE_RANGES, n_tilings=n_tilings, tiles_per_dim=tiles_per_dim)
        self.w = [[0.0] * self.tile_coder.n_features for _ in range(self.action_quantizer.n_buckets)]
        self.alpha = alpha / n_tilings
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.episodes_taken = 0

    def _state_vector(self, state: State) -> list:
        return [state.position, state.velocity, state.pole_angle, state.pole_angle_velocity]

    def _active(self, state: State) -> list:
        return self.tile_coder.get_active_features(self._state_vector(state))

    def _q_value(self, active: list, a_idx: int) -> float:
        row = self.w[a_idx]
        return sum(row[i] for i in active)

    def current_epsilon(self) -> float:
        return max(self.epsilon_min, self.epsilon_start * (1 - self.episodes_taken / self.epsilon_decay_episodes))

    def new_episode(self, state: State = None):
        self.episodes_taken += 1

    def get_action(self, state: State, training: bool = False) -> Action:
        epsilon = self.current_epsilon() if training else 0.0
        active = self._active(state)
        if random.random() < epsilon:
            action_index = random.randrange(self.action_quantizer.n_buckets)
        else:
            q_values = [self._q_value(active, a) for a in range(self.action_quantizer.n_buckets)]
            max_q = max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == max_q]
            action_index = random.choice(best_indices)
        return Action(self.action_quantizer.bucket_center(action_index))

    def update(self, state: State, action: Action, reward: float, next_state: State, next_action: Action,
               terminated: bool = False):
        active = self._active(state)
        a_idx = self.action_quantizer.discretize(action.get_velocity())
        q_sa = self._q_value(active, a_idx)
        if terminated:
            bootstrap = 0.0
        else:
            next_active = self._active(next_state)
            next_a_idx = self.action_quantizer.discretize(next_action.get_velocity())
            bootstrap = self.gamma * self._q_value(next_active, next_a_idx)
        td_error = reward + bootstrap - q_sa
        row = self.w[a_idx]
        for i in active:
            row[i] += self.alpha * td_error

    def save(self, dirname: str):
        import os
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "policy.json"), "w") as f:
            json.dump({
                "w": self.w,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay_episodes": self.epsilon_decay_episodes,
            }, f)
        self.action_quantizer.save(os.path.join(dirname, "action_quantizer.json"))

    @classmethod
    def load(cls, dirname: str):
        import os
        with open(os.path.join(dirname, "policy.json"), "r") as f:
            data = json.load(f)
        action_quantizer = ActionQuantizer.load(os.path.join(dirname, "action_quantizer.json"))
        policy = cls(action_quantizer=action_quantizer, gamma=data["gamma"],
                     epsilon_start=data["epsilon_start"], epsilon_min=data["epsilon_min"],
                     epsilon_decay_episodes=data["epsilon_decay_episodes"])
        policy.alpha = data["alpha"]
        policy.w = data["w"]
        return policy
