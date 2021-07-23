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
with open('qcbm-opt-12q3l-basin-4ed829df-edc7-43b6-8a84-cc5143a5cf97_result.json') as f:
    data = json.load(f)

# Extract target/measured bitstring distribution and distance measure values.
distances = []
minimum_distances = []
bitstring_distributions = []

current_minimum = 100000
number_of_qubits = 9
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
        print(len(step["qcbm-optimization-results"]["history"]))
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
print(current_minimum)

fig = plt.figure(figsize=(16, 8))
x_range=[i for i in range(len(distances))]
plt.xlabel('Evaluation')
plt.ylabel('Clipped Log-Likelihood')
plt.legend(loc="upper right")
plt.plot(x_range,distances,color='green',marker='.')
plt.plot(x_range,minimum_distances,color='purple')
plt.hlines(
        y=exact_distance_value,
        xmin=0,
        xmax=x_range[-1],
        color="darkgreen",
        label="expected",
        alpha=0.8,
        linestyle="--",
    )
plt.legend(loc="upper right")
plt.show()