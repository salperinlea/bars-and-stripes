import json
import matplotlib

# matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.cbook import get_sample_data
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import entropy


def get_ordered_list_of_bitstrings(num_qubits):
    bitstrings = []
    for i in range(2 ** num_qubits):
        bitstring = "{0:b}".format(i)
        while len(bitstring) < num_qubits:
            bitstring = "0" + bitstring
        bitstrings.append(bitstring)
    return bitstrings


# Insert the path to your JSON file here
with open("workflow_result.json") as f:
    data = json.load(f)

# Extract target/measured bitstring distribution and distance measure values.
distances = []
minimum_distances = []
bitstring_distributions = []

current_minimum = 100000
number_of_qubits = 16
for step_id in data:
    step = data[step_id]
    # if step["stepName"] == "get-initial-parameters":
    #     number_of_qubits = int(
    #         eval(step["inputParam:ansatz-specs"])["number_of_qubits"]
    #     )
    ordered_bitstrings = get_ordered_list_of_bitstrings(number_of_qubits)
    if step["stepName"] == "get-bars-and-stripes-distribution":
        target_distribution = []
        for key in ordered_bitstrings:
            try:
                target_distribution.append(
                    step["distribution"]["bitstring_distribution"][key]
                )
            except:
                target_distribution.append(0)
        exact_distance_value = entropy(target_distribution)
        print(exact_distance_value)
    elif step["stepName"] == "optimize-circuit":
        for evaluation in step["qcbm-optimization-results"]["history"]:
            distances.append(evaluation["value"]["value"])
            current_minimum = min(current_minimum, evaluation["value"]["value"])
            minimum_distances.append(current_minimum)

            bitstring_dist = []
            for key in ordered_bitstrings:
                try:
                    bitstring_dist.append(
                        evaluation["artifacts"]["bitstring_distribution"][key]
                    )
                except:
                    bitstring_dist.append(0)
            bitstring_distributions.append(bitstring_dist)

fig = plt.figure(figsize=(16, 8))

evals = []
plotted_distances = []
plotted_min_distances = []
line_widths = []


def animate(i):
    evals.append(i)
    plotted_distances.append(distances[i])
    plotted_min_distances.append(minimum_distances[i])
    line_widths.append(1)
    fig.clear()
    fig.set(
        xlabel="Evaluation Index",
        ylabel="Clipped negative log-likelihood cost function",
    )
    fig.set_ylim([exact_distance_value - 0.1, exact_distance_value + 1.5])
    fig.scatter(
        evals, plotted_distances, color="green", linewidths=line_widths, marker="."
    )
    fig.hlines(
        y=exact_distance_value,
        xmin=0,
        xmax=evals[-1],
        color="darkgreen",
        label="expected",
        alpha=0.8,
        linestyle="--",
    )
    fig.legend(loc="upper right")
    fig.plot(evals, plotted_min_distances, color="purple", linewidth=2)

    return fig


anim = FuncAnimation(fig, animate, frames=700, interval=1, repeat=False)

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('qcbm_opt_700_iterations.mp4', writer=writer)

plt.show()
