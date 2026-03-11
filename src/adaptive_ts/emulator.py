from copy import deepcopy
from hetgpy import crnGP
import numpy as np

class BaseEmulator:
    '''
    Base class to create an emulator. Requires implemening a fit, predict, and sample method
    '''
    def __init__(self):
        return

    def fit(self,X,Y):
        raise NotImplementedError()
    def predict(self):
        raise NotImplementedError()
    def sample(self):
        raise NotImplementedError()
    def copy(self):
        '''
        Return a copy of the model
        '''
        newmodel = deepcopy(self)
        return newmodel
    
class _hetGPyEmulator(BaseEmulator):
    def __init__(self,covtype = "Gaussian", lower = None, upper = None, known = {}, init = {}):
        self.covtype = covtype
        self.lower = lower
        self.upper = upper
        self.known = known
        self.init = init

        self.base_model = None

    def fit(self,X,Y):
        self.model = self.base_model() # instantiate model object
        self.model.mle(X=X,Z=Y,covtype = self.covtype, lower = self.lower, upper = self.upper)
    
    def predict(self,X,xprime = None):
        preds = self.model.predict(x=X,xprime=xprime)
        return preds
    def sample(self,X,size = 1):
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
        return np.random.multivariate_normal(m,cov, size = size)

class crnGPEmulator(_hetGPyEmulator):
    def __init__(self,covtype="Gaussian", lower=None, upper=None, known={}, init={}):
        super().__init__(covtype, lower, upper, known, init)
        self.base_model = crnGP