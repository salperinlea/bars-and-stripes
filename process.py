import qe.sdk.v1 as qe
import csv
import pandas as pd
import numpy as np
from scipy.stats import entropy
import zquantum.core
from zquantum.qcbm.ansatz import QCBMAnsatz
import zquantum.optimizers


def get_ordered_list_of_bitstrings(num_qubits):
    bitstrings = []
    for i in range(2 ** num_qubits):
        bitstring = "{0:b}".format(i)
        while len(bitstring) < num_qubits:
            bitstring = "0" + bitstring
        bitstrings.append(bitstring)
    return bitstrings

parcounts={}
for i in [3,4,6]:
    for j in ['all','star','line']:
        ans=QCBMAnsatz(number_of_qubits=12,number_of_layers=i,topology=j)
        count=ans.number_of_params
        string="{}-{}".format(str(i),j)
        parcounts[string]=count

