import qe.sdk.v1 as qe
import numpy as np
import typing
from typing import List, Optional, Union
import json

from zquantum.core.utils import create_object
from zquantum.core.interfaces.ansatz import Ansatz
import zquantum.core.bitstring_distribution
from zquantum.core.bitstring_distribution import BitstringDistribution
from zquantum.core.typing import Specs
from zquantum.core.utils import load_from_specs
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.qcbm.cost_function import QCBMCostFunction
from qequlacs import QulacsSimulator
from target import bars_and_stripes_zigzag

import random

def get_rc(n):
    assert isinstance(n,int)
    root = np.sqrt(n)
    if int(root) == root:
        return root,root
    else:
        for i in range(int(root),0,-1):
            co,rem = divmod(n,i)
            if rem == 0:
                return co,i
        return n,1



@qe.step(
    resource_def=qe.ResourceDefinition(
        cpu="1000m",
        memory="1000Mi",
        disk="1Gi",
    ),
)
def get_specs(method: str,options: dict):
    specs={}
    if method in ['adam',
            'adagrad',
            'adamax',
            'nadam',
            'sgd',
            'momentum',
            'nesterov',
            'rmsprop',
            'rmsprop-nesterov']:

        specs['method']=method
        specs['options']=options
        specs['module_name']="zquantum.optimizers.gd_optimizer"
        specs['function_name']='GDOptimizer'

    elif method in ['cobyla''l-bfgs-b','bfgs']:
        specs['module_name']="zquantum.optimizers.scipy_optimizer"
        specs['function_name']='ScipyOptimizer'
        specs['options']=options
    elif method[0:6] == 'basin-':
        minimizer_kwargs={}
        specialized=method[6:]
        minimizer_kwargs['method'] = specialized
        for k,v in options.items():
            if k == 'minimizer_kwargs':
                for q,p in v.keys():
                    minimizer_kwargs[q]=p
            else:
                specs[k]=v
        specs['minimizer_kwargs']=minimizer_kwargs
        specs['module_name']='zquantum.optimizers.basin_hopping'
        specs['function_name']='BasinHoppingOptimizer'
    return specs

@qe.step(
    resource_def=qe.ResourceDefinition(
        cpu="1000m",
        memory="1000Mi",
        disk="1Gi",
    ),
)
def get_ansatz(n_qubits:int,n_layers:int):
    return QCBMAnsatz(n_layers,n_qubits,'all')


@qe.step(
    resource_def=qe.ResourceDefinition(
        cpu="1000m",
        memory="1000Mi",
        disk="1Gi",
    ),
)
def generate_random_ansatz_params(
    ansatz,
    number_of_parameters: Optional[int] = None,
    min_value: float = -np.pi * 0.5,
    max_value: float = np.pi * 0.5,
    seed: int = None,
):

    if ansatz is not None:
        number_of_parameters = ansatz.number_of_params

    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(min_value, max_value, number_of_parameters)
    return params

@qe.step(
    resource_def=qe.ResourceDefinition(
        cpu="1000m",
        memory="1000Mi",
        disk="1Gi",
    ),
)
def get_distribution(n: int):
    row,col = get_rc(n)
    data = bars_and_stripes_zigzag(row, col)
    num_desired_patterns = int(len(data))
    num_desired_patterns = max(num_desired_patterns, 1)
    data = random.sample(list(data), num_desired_patterns)

    distribution_dict = {}
    for pattern in data:
        bitstring = ""
        for qubit in pattern:
            bitstring += str(qubit)

        distribution_dict[bitstring] = 1.
    return BitstringDistribution(distribution_dict)


@qe.step(
    resource_def=qe.ResourceDefinition(
        cpu="1000m",
        memory="1000Mi",
        disk="1Gi",
    ),
)
def optimize_variational_qcbm_circuit(
    ansatz,
    optimizer_specs,
    initial_parameters,
    target_distribution,
    keep_history: bool,
):
    backend = QulacsSimulator()
    if isinstance(optimizer_specs, str):
        optimizer_specs = json.loads(optimizer_specs)
    optimizer = create_object(optimizer_specs)
    cost_function = QCBMCostFunction(
        ansatz,
        backend,
        zquantum.core.bitstring_distribution.compute_clipped_negative_log_likelihood,
        {"epsilon": 1e-6},
        target_distribution,
    )
    opt_results = optimizer.minimize(cost_function, initial_parameters, keep_history)
    return opt_results
@qe.workflow(name='top40-{method}-{tag}',
              import_defs=[
                  qe.GitImportDefinition.get_current_repo_and_branch(),
                  qe.GitImportDefinition(
                      repo_url="git@github.com:zapatacomputing/z-quantum-core.git",
                      branch_name="gd_opt_hotfix",
                  ),
                  qe.GitImportDefinition(
                      repo_url="git@github.com:zapatacomputing/z-quantum-optimizers.git",
                      branch_name="gd_opt_troubleshoot",
                  ),
                  qe.GitImportDefinition(
                      repo_url="git@github.com:zapatacomputing/z-quantum-qcbm.git",
                      branch_name="master",
                  ),
                  qe.GitImportDefinition(
                      repo_url="git@github.com:zapatacomputing/qe-qulacs.git",
                      branch_name="master",
                  )

              ])
def workflow(n_layers: int, n_qubits: int,method: str,options: dict, keep_history: bool = True, tag: str = None):
    target_distribution=get_distribution(n_qubits)
    ansatz=get_ansatz(n_qubits,n_layers)
    initial_parameters=generate_random_ansatz_params(ansatz)
    optimizer_specs=get_specs(method,options)
    output = optimize_variational_qcbm_circuit(ansatz,optimizer_specs,
                                               initial_parameters,target_distribution,
                                               keep_history)
    return output

if __name__ == "__main__":
    n_layers=3
    n_qubits=12
    method,options ='adam',{'lr':0.001,'maxiter':5,'tol':1e-4}
    wf = workflow(n_layers,n_qubits,method,options,tag='a')
    #wf.validate()
    #qe.run_local_workflow(wf)
    out= wf.submit()
    out.watch_and_get_result()