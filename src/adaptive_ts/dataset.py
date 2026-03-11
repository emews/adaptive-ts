'''Module for dataset -- a wrapper around a dataframe'''
import pandas as pd
import numpy as np


def _identity(Y: np.ndarray):
    '''
    Returns the same input and output
    '''
    return Y
def _check_seed(X: np.ndarray):
    if not np.allclose(X[:,-1],X[:,-1].astype(int)):
        raise ValueError('Seed must be last column and encoded or convertible to integer')

class Dataset:
    '''
    Basic class to manage data handling for adaptive workflow. Essentially a dataframe (self.data) with some commands for appending 
    '''
    def __init__(self,X,Y,check_seed = False, transform = None):
        self.ym = None
        self.ystd = None

        if check_seed:
            _check_seed(X)
        if transform is None:
            transform = self._log_and_standardize
        self.transform = transform
        if isinstance(X,pd.DataFrame):
            self.data = X.copy()
            self.xcols = X.columns.tolist()
        else:
            self.data = pd.DataFrame(X,columns=[f"x{i}" for i in range(X.shape[1])])
            self.xcols = self.data.columns.tolist()
        self.data['Y_native'] = Y.copy()
        y= self.transform(Y,save_stats = True)
        self.data['Y'] = y 
        self.X = self.data[self.xcols].values
        self.Y = self.data['Y'].values
    def _log_and_standardize(self,Y: np.ndarray,save_stats = False):
        '''
        Default transform: take log of Y for numerical stability and standardize (Y - mean(Y)) / std(Y)

        This is runs at initialization with return_stats = True to also save the mean and standard deviation of y so it can be used
        when acquiring new observations
        
        Parameters
        ----------
        Y: np.ndarray
            array of observations
        save_stats: bool
            whether to update mean/standard deviation of y (self.ym, self.ystd)
        '''
        y = np.log(Y)
        if save_stats:
            self.ym, self.ystd = y.mean(), y.std()
        y = (y - self.ym) / self.ystd  
        return y
    def append(self,X,Y,add_sim_id = True):
        df = pd.DataFrame(X,columns=self.xcols)
        df['Y_native'] = Y
        df['Y'] = self.transform(Y)
        self.data = pd.concat([self.data,df],axis=0)
        if add_sim_id:
            self.data['sim_id'] = np.arange(1,len(self.data)+1)
        self.X = self.data[self.xcols].values
        self.Y = self.data['Y'].values
        return self

        

