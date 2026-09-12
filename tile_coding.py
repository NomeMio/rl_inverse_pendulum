import random


class TileCoder:
    """
    Multiple-overlapping-tilings feature representation for a continuous
    state space. Each tiling covers the full range of every dimension with
    `tiles_per_dim` equal-width tiles, offset by a random fraction of a
    tile width so that the tilings overlap rather than coincide. A state
    always activates exactly one tile per tiling, so `get_active_features`
    returns `n_tilings` indices into a sparse binary feature space of size
    `n_tilings * tiles_per_dim**len(ranges)`.
    """

    def __init__(self, ranges: list, n_tilings: int = 8, tiles_per_dim: int = 6, seed: int = 0):
        self.ranges = ranges
        self.n_dims = len(ranges)
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.tiles_per_tiling = tiles_per_dim ** self.n_dims
        self.n_features = n_tilings * self.tiles_per_tiling

        rng = random.Random(seed)
        widths = [(high - low) / tiles_per_dim for low, high in ranges]
        self.offsets = [
            [rng.uniform(0.0, w) for w in widths]
            for _ in range(n_tilings)
        ]

    def _tile_index(self, state_vector: list, tiling: int) -> int:
        index = 0
        for d, (low, high) in enumerate(self.ranges):
            width = (high - low) / self.tiles_per_dim
            value = min(max(state_vector[d], low), high) + self.offsets[tiling][d]
            bucket = int((value - low) / width)
            bucket = min(bucket, self.tiles_per_dim - 1)
            index = index * self.tiles_per_dim + bucket
        return index

    def get_active_features(self, state_vector: list) -> list:
        return [
            tiling * self.tiles_per_tiling + self._tile_index(state_vector, tiling)
            for tiling in range(self.n_tilings)
        ]
