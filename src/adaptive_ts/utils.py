import numpy as np
from itertools import product
def make_grid(*args):
    r'''
    
    Parameters
    ----------
    *args: iterable
        iterable of inputs. must all be ndarray_like and have two dimensions
    
    Returns
    -------
    np.array of cartesian product of all the entries in *args
    '''
    return np.array(list(map(np.concatenate,list(product(*args)))))