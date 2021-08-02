import qe.sdk.v1 as qe
import csv
import pandas as pd
import numpy as np
from scipy.stats import entropy
import zquantum.core
import zquantum.qcbm as qcbm
import zquantum.optimizers

def get_ordered_list_of_bitstrings(num_qubits):
    bitstrings = []
    for i in range(2 ** num_qubits):
        bitstring = "{0:b}".format(i)
        while len(bitstring) < num_qubits:
            bitstring = "0" + bitstring
        bitstrings.append(bitstring)
    return bitstrings
