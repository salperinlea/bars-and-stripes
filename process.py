import qe.sdk.v1 as qe
import csv
import pandas as pd
import numpy as np
from scipy.stats import entropy
import zquantum.core
from zquantum.qcbm.ansatz import QCBMAnsatz
import qeqiskit
import qiskit
from zquantum.core.circuits import export_to_qiskit
import zquantum.optimizers

def get_ordered_list_of_bitstrings(num_qubits):
    """
    function which gets a list of bitstrings. This is needed to check what the lowest possible loss is.
    :param num_qubits: int:
        number of qubits.
    :return:
        list of strings, each something like "0110011"
    """
    bitstrings = []
    for i in range(2 ** num_qubits):
        bitstring = "{0:b}".format(i)
        while len(bitstring) < num_qubits:
            bitstring = "0" + bitstring
        bitstrings.append(bitstring)
    return bitstrings

def get_best(id):
    try:
        data=qe.load_workflowresult(id)
        for step_id in data.steps:
            step = data.steps[step_id]
            if any([x in step_id for x in ["optimize-variational-qcbm-circuit", 'optimize-circuit']]):
                return step.result.opt_value
    except:
        return None

def get_convergence(id,tol):
    distances=[]
    try:
        data=qe.load_workflowresult(id)
        for step_id in data.steps:
            step = data.steps[step_id]
            if any([x in step_id for x in ["optimize-variational-qcbm-circuit", 'optimize-circuit']]):
                for evaluation in step.result.history:
                    distances.append(evaluation.value)
        min_d=min(distances)
        for i,dist in enumerate(distances):
            if abs(dist-min_d)<tol:
                return 3*i
    except:
        return None

#calculate how many parameters, moments, etc. are in each ansatz.

methods=['rmsprop','adam','sgd','l-bfgs-b','bfgs','cobyla']
strings=["{}-{}-{}".format(i,j,k) for i in[3,4,6] for j in ["all","star","line"] for k in methods]
counts={}
for i in [3,4,6]:
    for j in ['all','star','line']:
        subdict={}
        ans=QCBMAnsatz(number_of_qubits=12,number_of_layers=i,topology=j)
        count=ans.number_of_params
        subdict['n_params']=count

        exec= QCBMAnsatz.get_executable_circuit(params=np.ones(count))
        export= export_to_qiskit(exec)
        depth=export.depth()
        subdict['depth'] = depth
        size=export.size()
        entangles=size-count
        subdict['n_entanglers']=entangles
        for k in methods:
            string = "{}-{}-{}".format(str(i),j,k)
            counts[string]=subdict


filename='top20_ids.csv'
df=pd.read_csv(filename)

tol=1e-2 # set a convergence tolerance. play with this
df['performance'] = df.apply(lambda x: get_best(x.id),axis=1)
df['convergence'] = df.apply(lambda x: get_convergence(x.id,tol),axis=1)

new_file='top20_ids_data.csv'
df.to_csv(new_file) ## save this; the steps afterward are just data manipulation, not storage

performances={string:[] for string in strings}
convergences={string:[] for string in strings}
averages={k:{'performance':None,'convergence':None} for k in strings}
standards={k:{'performance':None,'convergence':None} for k in strings}
for row in df:
    string="{}-{}-{}".format(row.n_layers,row.topoloy,row.method)
    if row.performance is not None:
        performances[string].append(row.performance)
    if row.convergence is not None:
        convergences[string].append(row.convergence)

for k,v in performances.items():
    averages[k]['performance']=np.average(v)
    standards[k]['performance']=np.std(v)
    averages[k]['convergence']=np.average(convergences[k])
    standards[k]['convergence']=np.std(convergences[k])

new=pd.DataFrame(columns=['n_layers','topology','method'])
for i in [3,4,6]:
    for j in ['all','star','line']:
        for k in methods:
            new.append([i,j,k])

def get_average(x,k):
    string = "{}-{}-{}".format(x.n_layers,x.topology,x.method)
    return averages[string][k]

def get_std(x,k):
    string = "{}-{}-{}".format(x.n_layers,x.topology,x.method)
    return standards[string][k]


new['performance']=new.apply(lambda x: get_average(x,'performance'),axis=1)
new['p_std']=new.apply(lambda x: get_std(x,'performance'),axis=1)
new['convergence']=new.apply(lambda x: get_average(x,'convergence'),axis=1)
new['c_std']=new.apply(lambda x: get_std(x,'convergence'),axis=1)

def get_n_calls(x):
    string = "{}-{}-{}".format(x.n_layers,x.topology,x.method)
    if x.method == 'cobyla':
        calls = x.convergence
    else:
        calls= x.convergence + x.convergence*counts[string]['n_params']
    return calls

def get_cost(x):
    string = "{}-{}-{}".format(x.n_layers,x.topology,x.method)
    if x.method == 'cobyla':
        calls = x.convergence
    else:
        calls= x.convergence + x.convergence*counts[string]['n_params']
    return calls*counts[string]['depth']

new['evaluations']=new.apply(get_n_calls,axis=1)
new['cost']=new.apply(get_n_calls,axis=1)

new.to_csv('test.csv')
#new['performance_per_epoch']
#new['performance_per_quantum_op']


