import numpy
from matplotlib import pyplot


def func(x):
    return 2 * numpy.sin(x) + x**0.8

rng = numpy.random.default_rng(0)
n_train = 50
keys = numpy.sort(rng.random(n_train) * 5)
values = func(keys) + rng.normal(0.0, 0.5, (n_train,))


def compute_attention_weights(query, keys, sigma):
    # Compute distance-based weights
    dist = (query - keys) ** 2
    weights = numpy.exp(-dist / (2 * sigma**2))
    return weights / weights.sum()


def simple_attention(x, sigma=1.0):
    n = len(x)
    result = numpy.zeros(n)
    attention_history = []
    for i in range(n):
        attention_weights = compute_attention_weights(x[i], keys, sigma)
        result[i] = numpy.sum(attention_weights * values)
        attention_history.append(attention_weights)
    return result, attention_history


queries = numpy.arange(0, 5, 0.05)
result, attn_matrix = simple_attention(queries, sigma=0.5)

pyplot.plot(keys, values, "o", alpha=0.5)
pyplot.plot(queries, result)
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.show()

pyplot.imshow(attn_matrix, aspect="auto", origin="lower",
              extent=[keys[0], keys[-1], queries[0], queries[-1]],
              cmap="viridis")
pyplot.colorbar(label="Attention Weight")
pyplot.xlabel("Key (x position)")
pyplot.ylabel("Query (x position)")
pyplot.title("Attention Weight Heatmap")
pyplot.tight_layout()
pyplot.show()
