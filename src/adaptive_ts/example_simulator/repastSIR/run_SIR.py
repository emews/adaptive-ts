'''
Set of functions to run the SIR model created in SIR.py. Upon running this script from the command line, it will:

1. create `df_pars`, a pd.DataFrame where each row 


'''



import yaml
import pandas as pd
import numpy as np
from itertools import product
from adaptive_ts.example_simulator.repastSIR.SIR import run
from time import time
from joblib import Parallel, delayed
import os
import sqlite3 as sql
from tqdm import tqdm


COL_DTYPES = {'init_infect_loc':str}

def read_yaml(path: str) -> dict:
    r'''
    Reads in a yaml file to a dict
    '''
    with open(path,'rb') as stream:
        return yaml.safe_load(stream)
    
def dict_product(input_dict: dict):
    '''
    Makes cartesian product of list values for each key. Copied from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists/40623158#40623158 
    
    Example
    -------
    >>> list(dict_product({'number': [1, 2], 'character': 'ab'}))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    '''
    # validate inputs
    for key, val in input_dict.items():
        if type(val) != list:
            input_dict[key] = [val]

    return (dict(zip(input_dict.keys(), x)) for x in product(*input_dict.values()))
def setup_experiment(experiment_path: str,varying_pars: dict) -> pd.DataFrame:
    r'''
    
    Parameters
    ----------
    experiment_path: str
        filepath where pars.yaml lives
    varying_pars: dict
        dictionary of pars that vary across experiments (in this case, beta and seed)

    Returns
    -------
    df_pars: pd.DataFrame
        dataframe containing cartesian product of varying model parameters combined with constant parameters. Also contains a unique run number per row
    '''
    base_pars = read_yaml(experiment_path)
    pars = base_pars.copy()
    
    for key, val in varying_pars.items():
        pars[key] = val

    df_pars = pd.DataFrame(dict_product(pars))
    df_pars['run_number'] = range(1,len(df_pars)+1)
    df_pars['counts_file'] = [f"output/data_{i:05d}.csv" for i in df_pars['run_number'].values]
    return df_pars

def split_dataframe(df,n_chunks) -> list:
    r'''
    Splits dataframe into list of dataframes. Used for batch processing

    Parameters
    ----------
    df: pd.DataFrame
        dataframe to split
    n_chunks: int
        number of chunks to create
    
    Returns
    -------
    split: list
        list of dataframes
    '''
    if n_chunks == 1:
        return [df]
    chunk_size = int(len(df) / n_chunks)
    split = []
    for start in range(0,len(df),chunk_size):
        df_split = df.iloc[start:start + chunk_size].copy()
        split.append(df_split)
    return split



def run_batch(df_pars: pd.DataFrame) -> str:
    r'''
    Run a batch of simulations
    assumes that pars["return_data"] is True, so saves df of results (to save on IO over many experiments, since running SIR.py is cheap (~3 sec per model run))
    
    Parameters
    ----------
    df_pars: pd.DataFrame
        ataframe where each row is a model parameterization. Each row contains a unique run_number

    Returns
    -------
    fp: str
        filepath where output of run(pars) is stored, labled as "data_[minimum run number]_to_[maximum run number].csv"
    '''
    results = []
    for i, row in df_pars.iterrows():
        pars = dict(row)
        out = run(pars)
        # post-process
        for key in ['run_number','beta','gamma','seed']:
            out[key] = pars[key]
        results.append(out)
    results = pd.concat(results)
    from_rn = results['run_number'].min()
    to_rn = results['run_number'].max()
    fp = f'output/data_{from_rn:06d}_to_{to_rn:06d}.csv'
    #results.to_csv(fp,index=False)
    return results

def run_batch(X,ytrue):
    Y = []
    sims = []
    for i, par in enumerate(X):
        res = run_sim_err(par,y_true=ytrue)
        sim = res['res']
        sim['sim_id'] = i
        sims.append(sim)
        Y.append(res['out'])
    sims = pd.concat(sims)
    Y = np.array(Y)
    return {'Y':Y,'sims':sims}
            
def run_experiment_sequential(df_pars: pd.DataFrame,n_chunks: int = 2) -> None:
    r'''
    The single core (sequential) implementation of run_experiment_parallel
    '''
    batches = split_dataframe(df_pars,n_chunks)
    results = []
    for batch in batches:
        output = run_batch(batch)
        results.append(output)
    return pd.concat(results)
def run_experiment_parallel(df_pars: pd.DataFrame,n_chunks: int = 2,n_cpus: int = 2,return_as: str="list") -> list:
    r'''
    Runs an experiment in parallel using joblib

    Parameters
    ----------
    df_pars: pd.DataFrame
        dataframe where each row is a model parameterization. Each row contains a unique run_number
    n_chunks: int
        number of equal chunks to split df_pars into
    n_cpus: int
        number of cores to use for joblib
    return_as: str
        type of joblib output to use, list or generator (does not make a difference for small experiments)
    Returns
    -------
    output: generator or list of filepaths containing path to csv of output results 
        (each csv is labeled as as data_[starting_run_number]_to_[ending_run_number].csv)
    
    '''
    batches = split_dataframe(df_pars,n_chunks)
    parallel = Parallel(n_jobs=n_cpus,return_as=return_as)
    output =  parallel(delayed(run_batch)(batch) for batch in batches)
    return output

def remove_old_files() -> None:
    r'''
    Removes old files from the root of the output directory, which is used as a temp directory 
    (useful if a model run fails)

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    for file in os.listdir('output'):
        file = 'output/' + file
        if file.endswith('.csv'):
            os.remove(file)

def collect_and_write_to_own_folder(new_dir: str,outputs: list) -> None:
    r'''
    Collect batched outputs from model runs and 

    Parameters
    ----------
    new_dir: str
        directory to write outputs to
    outputs: list
        list of filepaths containing batched model outputs 
    Returns
    -------
    None
    
    '''
    if 'output' not in new_dir:
        new_dir = 'output/' + new_dir
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    # gather csvs
    outputs = [os.path.basename(file) for file in outputs]
    for file in outputs:
        file = 'output/' + file
        if file.endswith('.csv'):
            os.rename(file,f"{new_dir}/" + os.path.basename(file))

def collect_and_write_to_db(results, db_path: str):
    start = time()
    with sql.connect(db_path) as conn:
        for result in tqdm(results):
            result.to_sql("data",conn,if_exists='append')
        end = time()
        df_time = pd.DataFrame([{'time':end-start}])
        df_time.to_sql('time',conn)

if __name__ == "__main__":

    varying_pars = {
        'seed': list(range(20)),
        'beta': np.linspace(0.03,0.15,50).tolist(),
        'override_seed': [54],
        'save_movement': [False]
    }
    fixed_pars = 'input/pars_box_norm_ground_truth.yaml'
    df_pars_ground_truth = setup_experiment(fixed_pars,varying_pars={})
    df_pars_ground_truth['save_movement'] = True
    results_ground_truth = run_experiment_sequential(df_pars=df_pars_ground_truth,n_chunks=1)
    

    df_pars = setup_experiment(fixed_pars,varying_pars=varying_pars)
    # cast columns
    experiment_name = 'experiment_SIR_box_norm_override_seed_54'

    print(f"Running experiment of size {len(df_pars)}")
    s = time()
    n_chunks, n_cpus = 20, 12
    results = run_experiment_parallel(df_pars,n_chunks=n_chunks,n_cpus=n_cpus,return_as="generator_unordered")

    db_path = f'output/{experiment_name}_size_{len(df_pars)}_chunks_{n_chunks}_cpus_{n_cpus}.db'
    if os.path.exists(db_path):
        print(f"Removing existing db")
        os.remove(db_path)
    with sql.connect(db_path) as conn:
        # remap col types to be agreeable with sqlite
        #results_ground_truth.to_sql('ground_truth',conn,if_exists='replace')
        df_pars.astype(COL_DTYPES).to_sql("pars",conn,if_exists='replace')
    
    collect_and_write_to_db(results,db_path=db_path)
    e = time()
    print(f"Experiment took: {e-s:0.02f} seconds")
    
