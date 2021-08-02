import json
import qe.sdk.v1 as qe
import matplotlib
import pickle
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
#with open('./results/top40-adam-b-cedb21be-7eca-4f3e-acbb-55703470f230_result.json') as f:
 #   data = json.load(f)

data=qe.load_workflowresult('top40-4-adam-b')
#print(data)
# Extract target/measured bitstring distribution and distance measure values.
distances = []
minimum_distances = []
bitstring_distributions = []

current_minimum = 100000
number_of_qubits = 12
for step_id in data.steps:
    step = data.steps[step_id]
    # if step["stepName"] == "get-initial-parameters":
    #     number_of_qubits = int(
    #         eval(step["inputParam:ansatz-specs"])["number_of_qubits"]
    #     )
    ordered_bitstrings = get_ordered_list_of_bitstrings(number_of_qubits)
    if "get-distribution" in step_id:
        target_distribution = []
        for key in ordered_bitstrings:
            try:
                target_distribution.append(
                    step.result.distribution_dict[key]
                )

            except:
                target_distribution.append(0)
        exact_distance_value = entropy(target_distribution)
        print("Entropy of true distribution: ",exact_distance_value)
    elif any([x in step_id for x in ["optimize-variational-qcbm-circuit", 'optimize-circuit']]):
        #print(type(step.result))
        #print(step.result.history)
        for evaluation in step.result.history:
            distances.append(evaluation.value)
            current_minimum = min(current_minimum, evaluation.value)
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
print("best performance by QCBM: ",current_minimum)

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