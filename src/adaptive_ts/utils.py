import numpy as np
from itertools import product
from scipy.stats import norm

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

def loglik(ysim, ytrue, err_sig):
    r'''
    Likelihood under the normal distribution

    Parameters
    ----------
    ysim: ndarray_like
        simulated outputs
    ytrue: float
        observed output
    err_sig: float
        variance of the normal distribution
    
    Returns
    -------
    log likelihood of the data given the simulation outputs
    '''
    return np.log(norm.pdf(ysim,ytrue,err_sig))

def loglik_design(model,newX,ytrue,err_sig):
    r'''
    Likelihood of the data given the predictions

    Parameters
    ----------
    model: hetgpy model
    newX: ndarray_like
        candidate designs (one row per observation)
    ytrue: float
        ground truth
    err_sig: float
        variance of ytrue distribution
    
    Returns
    -------
    ll: loglikelihoods at newX
    '''
    preds = model.predict(newX)

    ll = loglik(preds['mean'],ytrue,err_sig)
    return ll.squeeze()

def logit(x, eps=1e-6):
    r'''
    Maps data from [0,1] to the real line. Used so flow sampling does not have to be aware of bounds

    Parameters
    ----------
    x: ndarray_like
        Data
    eps: float
        clipping bounds for numerical stability
    
    Returns
    -------
    log(x/1-x)
    '''
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def inv_logit(z):
    r'''
    Inverse logit transformation. Map z from real line to [0,1]

    Parameters
    ----------
    z: ndarray_like

    Returns:

    1 / (1 + exp(-z))
    '''
    return 1 / (1 + np.exp(-z))