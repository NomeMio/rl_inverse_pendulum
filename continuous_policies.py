import json
import random
from math import exp

from cart_model import Action, State
from hyperparam_search import GridSearchMixin
from quantizer import ActionQuantizer
from tile_coding import TileCoder




class ReinforcePolicy(GridSearchMixin):
    """
    Discrete-action REINFORCE with a softmax-in-preferences policy, in two
    parameterizations selected by `weight_mode`:

    - "per_action" (default): a separate weight row per action, applied to
      state-only features -- preference(s,a) = weights[a] . x(s). This is
      the current/default mode.
    - "shared": a single weight vector shared across all actions, applied to
      a joint feature_fn(state, action) -- preference(s,a) = weights . x(s,a).
      This only works if x(s,a) actually varies with `a` (e.g. via state*action
      interaction terms): any feature component that doesn't depend on the
      action is identical across all actions for a given state, so it cancels
      out exactly in the softmax (shift-invariance) and its weight's gradient
      is mathematically zero -- it can never influence which action is
      chosen. The default `default_feature_fn_shared` is built entirely from
      such interaction terms for that reason; a custom feature_fn passed in
      "shared" mode should follow the same rule.
    """

    @staticmethod
    def default_feature_fn_per_action(state: State) -> list:
        """State-only features for "per_action" mode (each action already
        gets its own weight row, so the action doesn't need to appear here)."""
        return [1.0, state.position, state.velocity, state.pole_angle, state.pole_angle_velocity]

    @staticmethod
    def default_feature_fn_shared(state: State, action: Action) -> list:
        """Joint state*action interaction features for "shared" mode -- every
        term depends on the action, so none of them can cancel in the softmax
        (see class docstring)."""
        a = action.get_velocity()
        return [
            a,
            state.position*a,
            state.velocity*a, 
            state.pole_angle* a,
            state.pole_angle_velocity * a,
        ]

    @classmethod
    def default_param_grid(cls) -> dict:
        return {
            "alpha": [0.001, 0.005, 0.01],
            "discount": [0.95, 0.99],
        }

    def __init__(self, alpha: float = 0.005, discount: float = 0.99,
                action_limits=None,
                feature_fn=None,
                weight_mode: str = "per_action"):
        if weight_mode not in ("per_action", "shared"):
            raise ValueError(f"weight_mode must be 'per_action' or 'shared', got {weight_mode!r}")
        self.weight_mode = weight_mode
        self.alpha = alpha
        self.action_limits = list(action_limits) if action_limits is not None else \
            [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.num_actions = len(self.action_limits) - 1
        self.discount = discount
        probe_state = State([0.0, 0.0, 0.0, 0.0])
        if weight_mode == "per_action":
            self.feature_fn = feature_fn or self.default_feature_fn_per_action
            self.length_features = len(self.feature_fn(probe_state))
            self.weights = [[random.uniform(-1, 1) for _ in range(self.length_features)]
                             for _ in range(self.num_actions)]
        else:
            self.feature_fn = feature_fn or self.default_feature_fn_shared
            self.length_features = len(self.feature_fn(probe_state, Action(0.0)))
            self.weights = [random.uniform(-1, 1) for _ in range(self.length_features)]

    def _action_for_index(self, i: int) -> Action:
        return Action((self.action_limits[i] + self.action_limits[i + 1]) / 2.0)

    def _action_index(self, action: Action) -> int:
        """Nearest bucket center to `action`'s velocity -- robust to the
        exact float coming from anywhere (not just _action_for_index)."""
        target = action.get_velocity()
        best_i, best_d = 0, None
        for i in range(self.num_actions):
            center = (self.action_limits[i] + self.action_limits[i + 1]) / 2.0
            d = abs(center - target)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        return best_i

    def _preferences(self, state: State) -> list:
        if self.weight_mode == "per_action":
            features = self.feature_fn(state)
            return [sum(w * f for w, f in zip(row, features)) for row in self.weights]
        return [
            sum(w * f for w, f in zip(self.weights, self.feature_fn(state, self._action_for_index(i))))
            for i in range(self.num_actions)
        ]

    def _action_probs(self, state: State) -> list:
        preferences = self._preferences(state)
        max_pref = max(preferences)
        exp_prefs = [exp(p - max_pref) for p in preferences]
        total = sum(exp_prefs)
        return [e / total for e in exp_prefs]

    def new_episode(self, state: State = None):
        pass

    def get_action(self, state: State, training: bool = False) -> Action:
        probs = self._action_probs(state)
        if not training:
            best = max(range(self.num_actions), key=lambda i: probs[i])
            return self._action_for_index(best)
        uni = random.uniform(0, 1)
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if uni <= cumulative:
                return self._action_for_index(i)
        return self._action_for_index(self.num_actions - 1)

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
            a_idx = self._action_index(action)
            probs = self._action_probs(state)
            coeff = self.alpha * discount_pow * returns[t]
            if self.weight_mode == "per_action":
                # grad_{theta_b} ln pi(a|s) = (1[b==a] - pi(b|s)) * x(s)
                features = self.feature_fn(state)
                for b in range(self.num_actions):
                    grad_coeff = (1.0 if b == a_idx else 0.0) - probs[b]
                    row = self.weights[b]
                    for k in range(self.length_features):
                        row[k] += coeff * grad_coeff * features[k]
            else:
                # grad_theta ln pi(a|s) = x(s,a) - sum_b pi(b|s)*x(s,b)
                taken_features = self.feature_fn(state, self._action_for_index(a_idx))
                expected_features = [0.0] * self.length_features
                for b in range(self.num_actions):
                    f = self.feature_fn(state, self._action_for_index(b))
                    for k in range(self.length_features):
                        expected_features[k] += probs[b] * f[k]
                for k in range(self.length_features):
                    self.weights[k] += coeff * (taken_features[k] - expected_features[k])
            discount_pow *= self.discount

    def save(self, dirname: str):
        import os
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "policy.json"), "w") as f:
            json.dump({
                "weights": self.weights,
                "alpha": self.alpha,
                "discount": self.discount,
                "action_limits": self.action_limits,
                "weight_mode": self.weight_mode,
            }, f)

    @classmethod
    def load(cls, dirname: str, feature_fn=None):
        """`feature_fn` must be passed again if a non-default one was used to
        train the saved policy — it's a callable, so it isn't persisted in
        the JSON file."""
        import os
        with open(os.path.join(dirname, "policy.json"), "r") as f:
            data = json.load(f)
        policy = cls(alpha=data["alpha"], discount=data["discount"],
                     action_limits=data["action_limits"], feature_fn=feature_fn,
                     weight_mode=data.get("weight_mode", "per_action"))
        policy.weights = data["weights"]
        return policy
