from copy import deepcopy
from hetgpy import crnGP
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, Exponentiation, ExpSineSquared


class BaseEmulator:
    '''
    Base class to create an emulator. Requires implemening a fit, predict, and sample method
    '''
    def __init__(self,rng = None):
        self.rng = rng
        return
    def __str__(self):
        return 
    def fit(self,X,Y):
        raise NotImplementedError()
    def predict(self):
        raise NotImplementedError()
    def sample(self):
        raise NotImplementedError()
    def acquire(self,X):
        tTS = self.sample(X)
        # take the TS
    def copy(self):
        '''
        Return a copy of the model
        '''
        newmodel = deepcopy(self)
        return newmodel
    

class GaussianProcessEmulator(BaseEmulator):
    def __init__(self, kernel: kernels):
        super().__init__()
        self.kernel = kernel
        self.model = GaussianProcessRegressor(kernel=kernel)
    def fit(self,X,Y):
        # sklearn Gaussian Process ignores seed information
        if (X[:,-1] == X[:,-1].astype(int)).all():
            X = X[:,0:X.shape[1]-1]
            self.model.fit(X,Y)
    def predict(self,X,return_cov=False):
        if (X[:,-1] == X[:,-1].astype(int)).all():
            X = X[:,0:X.shape[1]-1]
        out = {}
        if return_cov:
            m, cov = self.model.predict(X=X,return_cov=True)
            out['mean'] = m
            out['cov'] = cov
        else:
            out['mean'] = self.model.predict(X=X)
        return out
    def sample(self,X,size=1):
        preds = self.predict(X=X,return_cov=True)
        m, cov = preds['mean'], preds['cov']
        cov = 0.5 * (cov + cov.T)
        eigval, eigvec = np.linalg.eigh(cov)
        if (eigval < 0).any():
            # threshold eigvalues to zero and re-calculate covariance matrix
            eigval[eigval < 0] = 0
            cov = eigvec @ np.diag(eigval) @ eigvec.T
        return self.rng.multivariate_normal(m,cov, size = size)
    


class _hetGPyEmulator(BaseEmulator):
    r'''
    Emulator wrapper for hetGPy models. Model hyperparameters (such as covariance structure, lengthscale limits, etc) must be specified at model instantiation
    '''
    def __init__(self,covtype = "Gaussian", lower = None, upper = None, known = {}, init = {}):
        self.covtype = covtype
        self.lower = lower
        self.upper = upper
        self.known = known
        self.init = init
        self.rng = None
        self.base_model = None

    def fit(self,X: np.ndarray,Y: np.ndarray):
        r'''
        Calls self.mle of hetGPy model

        Parameters
        ----------
        X: np.ndarray
            np.ndarray of n rows and d columns corresponding to model inputs
        Y: np.ndarray
            np.ndarray of outputs, must be same length as X

        Returns
        -------
        None
        '''
        self.model = self.base_model() # instantiate model object
        self.model.mle(X=X,Z=Y,covtype = self.covtype, lower = self.lower, upper = self.upper)
    
    def predict(self,X: np.ndarray,xprime: np.ndarray | None = None):
        r'''
        Wrapper around predict method of hetGPy module. 

        Parameters
        ----------
        X: np.ndarray
            np.ndarray of locations to predict at
        xprime: np.ndarray 
            inputs for returning covariance matrix

        Returns
        -------
        preds: dict
            dict with mean, sd2, nugs, and xprime
        '''
        preds = self.model.predict(x=X,xprime=xprime)
        return preds
    def sample(self,X,size = 1):
        r'''
        Sample from Gaussian process posterior

        Parameters
        ----------
        X: np.ndarray
            np.ndarray of locations to sample from
        size: int
            number of samples to acquire

        Returns
        -------
        np.ndarray matrix of (size rows x len(X)) corresponding to samples from GP posterior
        '''
        preds = self.model.predict(x=X,xprime=X)
        # check covariance matrix
        m = preds['mean']
        cov = preds['cov']
        cov = 0.5 * (cov + cov.T)
        eigval, eigvec = np.linalg.eigh(cov)
        if (eigval < 0).any():
            # threshold eigvalues to zero and re-calculate covariance matrix
            eigval[eigval < 0] = 0
            cov = eigvec @ np.diag(eigval) @ eigvec.T
        return self.rng.multivariate_normal(m,cov, size = size)

class crnGPEmulator(_hetGPyEmulator):
    r'''
    Wrapper for _hetGPyEmulator that applies hetgpy.crnGP as the emulator.
    '''
    def __init__(self,covtype="Gaussian", lower=None, upper=None, known={}, init={}):
        super().__init__(covtype, lower, upper, known, init)
        self.base_model = crnGP
    def __str__(self):
        super().__str__()
        return 'crnGP'