import cart_model
import random
from quantizer import StateQuantizer

random.seed(0)
context = cart_model.Cart_model(human=False)

names = ["position", "velocity", "pole_angle", "pole_angle_velocity"]


def print_distribution(title, quantizer, samples):
    total = len(samples["position"])
    print(f"\n=== {title} ({total} states total) ===")
    for name in names:
        bucketizer = quantizer.get_bucketizer(name)
        boundaries = [bucketizer.low] + bucketizer.edges + [bucketizer.high]
        counts = [0] * len(bucketizer)
        for value in samples[name]:
            counts[bucketizer.index(value)] += 1
        print(f"\n{name} distribution:")
        for idx, c in enumerate(counts):
            low, high = boundaries[idx], boundaries[idx + 1]
            pct = 100 * c / total if total else 0
            bar = "#" * int(pct)
            print(f"  bucket {idx:2d} [{low:7.3f}, {high:7.3f}): {c:7d} ({pct:5.2f}%) {bar}")


# collect raw scalar samples from 10000 episodes
samples = {name: [] for name in names}
for i in range(20000):
    episode = cart_model.Episode()
    while episode.terminated == False:
        state, action, reward = episode.one_step(context, None)
        samples["position"].append(state.position)
        samples["velocity"].append(state.velocity)
        samples["pole_angle"].append(state.pole_angle)
        samples["pole_angle_velocity"].append(state.pole_angle_velocity)

# equal-width buckets (original approach)
equal_width_quantizer = StateQuantizer(n_buckets=10)
print_distribution("Equal-width buckets", equal_width_quantizer, samples)

# quantile (equal-frequency) buckets, fitted on the same rollout data
quantile_quantizer = StateQuantizer.from_samples(samples, n_buckets=10)
print_distribution("Quantile (equal-frequency) buckets", quantile_quantizer, samples)

quantile_quantizer.save("saved_policies/state_quantizer.json")