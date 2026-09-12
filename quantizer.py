import bisect
import json
import math
import statistics

from cart_model import State


class Bucketizer:
    """
    Splits a continuous range [low, high] into buckets and maps a value
    to the index of the bucket it falls into.

    By default the range is split into `n_buckets` equal-width buckets,
    but the edges can be overridden (as a whole, or one bucket at a time)
    to make some buckets narrower/wider than others.
    """

    def __init__(self, low: float, high: float, n_buckets: int = 10, edges: list = None):
        self.low = low
        self.high = high
        if edges is not None:
            self.set_edges(edges)
        else:
            self.n_buckets = n_buckets
            width = (high - low) / n_buckets
            self.edges = [low + i * width for i in range(1, n_buckets)]

    def set_edges(self, edges: list):
        """Replace all internal boundaries at once. `edges` are the n_buckets-1
        interior cut points (strictly increasing, all within (low, high))."""
        self.edges = sorted(edges)
        self.n_buckets = len(self.edges) + 1

    def resize_bucket(self, bucket_index: int, new_right_edge: float):
        """Move the right boundary of a single bucket, keeping every other
        boundary fixed. Lets you shrink/grow one bucket without touching
        the equal-division logic used for the rest."""
        if bucket_index < 0 or bucket_index >= self.n_buckets - 1:
            raise IndexError(f"bucket_index must be in [0, {self.n_buckets - 2}]")
        self.edges[bucket_index] = new_right_edge
        self.edges.sort()

    def index(self, value: float) -> int:
        value = min(max(value, self.low), self.high)
        return bisect.bisect_right(self.edges, value)

    def center(self, index: int) -> float:
        """Midpoint of the given bucket, used to map a discrete action
        index back to a representative continuous value."""
        boundaries = [self.low] + self.edges + [self.high]
        return (boundaries[index] + boundaries[index + 1]) / 2

    def __len__(self):
        return self.n_buckets

    @classmethod
    def from_samples(cls, samples: list, n_buckets: int = 10, low: float = None, high: float = None):
        """Build a Bucketizer whose interior edges are placed at the
        n_buckets-1 quantiles of `samples`, so each bucket ends up covering
        roughly the same number of observations (equal-frequency binning)
        instead of an equal slice of the value range."""
        samples = sorted(samples)
        if low is None:
            low = samples[0]
        if high is None:
            high = samples[-1]
        edges = statistics.quantiles(samples, n=n_buckets, method="inclusive")
        edges = [min(max(e, low), high) for e in edges]
        return cls(low, high, edges=edges)


class StateQuantizer:
    """
    Turns a continuous cart_model.State into a tuple of bucket indices,
    one per state variable. Each variable has its own Bucketizer so
    ranges/bucket counts/individual bucket widths can be tuned independently.

    Default ranges are taken from the edge values measured in
    test_space_edge_values.py, widened slightly to be safe.
    """

    DEFAULT_RANGES = {
        "position": (-0.4, 0.4),
        "velocity": (-4.0, 4.0),
        "pole_angle": (-0.5, 0.5),
        "pole_angle_velocity": (-7.0, 7.0),
    }

    def __init__(self, n_buckets: int = 10, ranges: dict = None):
        ranges = ranges or self.DEFAULT_RANGES
        self.bucketizers = {}
        for name, (low, high) in ranges.items():
            self.bucketizers[name] = Bucketizer(low, high, n_buckets)

    @classmethod
    def from_samples(cls, samples: dict, n_buckets: int = 10, ranges: dict = None):
        """Build a StateQuantizer with quantile (equal-frequency) buckets,
        fitted on observed state-variable samples collected from rollouts.
        `samples` maps variable name -> list of observed scalar values."""
        ranges = ranges or cls.DEFAULT_RANGES
        quantizer = cls.__new__(cls)
        quantizer.bucketizers = {}
        for name, (low, high) in ranges.items():
            quantizer.bucketizers[name] = Bucketizer.from_samples(
                samples[name], n_buckets, low=low, high=high
            )
        return quantizer

    def get_bucketizer(self, name: str) -> Bucketizer:
        return self.bucketizers[name]

    def save(self, filename: str = "saved_policies/state_quantizer.json"):
        """Persist the fitted edges so they don't need to be recomputed
        (refitting quantile edges means re-running thousands of episodes)."""
        data = {
            name: {"low": b.low, "high": b.high, "edges": b.edges}
            for name, b in self.bucketizers.items()
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename: str = "saved_policies/state_quantizer.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        quantizer = cls.__new__(cls)
        quantizer.bucketizers = {
            name: Bucketizer(d["low"], d["high"], edges=d["edges"])
            for name, d in data.items()
        }
        return quantizer

    def discretize(self, state: State) -> tuple:
        return (
            self.bucketizers["position"].index(state.position),
            self.bucketizers["velocity"].index(state.velocity),
            self.bucketizers["pole_angle"].index(state.pole_angle),
            self.bucketizers["pole_angle_velocity"].index(state.pole_angle_velocity),
        )
    def getEdgesFromBucketizer(self, name: str) -> list:
        """Return the edges of the bucketizer for a given state variable."""
        return self.bucketizers[name].edges

    def n_states(self) -> int:
        n = 1
        for b in self.bucketizers.values():
            n *= len(b)
        return n


class ActionQuantizer:
    """
    Quantizes a single continuous action dimension (imprinted velocity)
    into a bucket index, using the same Bucketizer machinery as
    StateQuantizer. Supports equal-width buckets by default and
    quantile (equal-frequency) rebalancing via from_samples.
    """

    def __init__(self, low: float = -3.0, high: float = 3.0, n_buckets: int = 10, edges: list = None):
        self.bucketizer = Bucketizer(low, high, n_buckets, edges=edges)

    @property
    def n_buckets(self) -> int:
        return self.bucketizer.n_buckets

    @classmethod
    def from_samples(cls, samples: list, n_buckets: int = 10, low: float = -3.0, high: float = 3.0):
        quantizer = cls.__new__(cls)
        quantizer.bucketizer = Bucketizer.from_samples(samples, n_buckets, low=low, high=high)
        return quantizer

    @classmethod
    def focused_near_zero(cls, low: float = -3.0, high: float = 3.0, n_buckets: int = 10, power: float = 2.0):
        """Non-linear edges that concentrate resolution near 0 and get
        coarser towards the extremes, by warping equally-spaced cut
        points through |u|**power (power > 1 bunches edges near 0)."""
        edges = []
        for i in range(1, n_buckets):
            u = -1.0 + 2.0 * i / n_buckets
            warped = math.copysign(abs(u) ** power, u)
            edges.append(warped * high if warped >= 0 else warped * (-low))
        return cls(low, high, n_buckets, edges=edges)

    def discretize(self, velocity: float) -> int:
        return self.bucketizer.index(velocity)

    def bucket_center(self, index: int) -> float:
        return self.bucketizer.center(index)

    def save(self, filename: str = "saved_policies/action_quantizer.json"):
        data = {"low": self.bucketizer.low, "high": self.bucketizer.high, "edges": self.bucketizer.edges}
        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename: str = "saved_policies/action_quantizer.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        quantizer = cls.__new__(cls)
        quantizer.bucketizer = Bucketizer(data["low"], data["high"], edges=data["edges"])
        return quantizer
