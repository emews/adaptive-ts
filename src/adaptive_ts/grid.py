from time import time
import numpy as np
from scipy.stats import qmc
from adaptive_ts.utils import make_grid
from scipy.stats import norm
class BaseGridStrategy:
    def __init__(self, ndim:int = None, nseeds:int = None, rng = None):
        self.ndim = ndim
        self.nseeds = nseeds

        # validate random number generator
        if rng is None:
            seed = time()
            rng = np.random.default_rng(seed)
        if isinstance(rng,int):
            rng = np.random.default_rng(seed)
        elif isinstance(rng,np.random.Generator):
            pass
        else:
            raise ValueError('rng must be either int or numpy.random.Generator')
        self.rng = rng
        # set Xgrid
        self.Xgrid = None
    def sample(self):
        raise NotImplementedError()

class FixedGridStrategy(BaseGridStrategy):
    def __init__(self,ndim: int = None,nseeds:int = None, rng = None):
        super().__init__(ndim=ndim, nseeds=nseeds, rng=rng)
    
    def sample(self,n):
        if self.Xgrid is not None:
            return self.Xgrid
        else:
            X = qmc.LatinHypercube(d=self.ndim,rng=self.rng).random(n=n)
            seeds = (np.arange(self.nseeds) + 1).reshape(-1,1)

            self.Xgrid = make_grid(X,seeds)
            return self.Xgrid

class LatinHypercubeStrategy(BaseGridStrategy):
    def __init__(self,ndim: int = None,nseeds:int = None, rng = None,ngrid = None,nTS_samp = 30):
        super().__init__(ndim=ndim, nseeds=nseeds, rng=rng)
        self.lhs = qmc.LatinHypercube(d=self.ndim,rng=self.rng)
        self.ngrid = ngrid
        self.nTS_samp = nTS_samp
    def sample(self,model):
        X = self.lhs.random(self.ngrid)
        seeds = (np.arange(self.nseeds)+1).reshape(-1,1)
        self.Xgrid = make_grid(X,seeds)
        # Thompson samples
        tTS = model.sample(self.Xgrid,size=self.nTS_samp)
        best_ids = np.argmin(tTS,axis=1)
        best_ids = np.unique(best_ids,axis=0)       
        return self.Xgrid[best_ids]



class AdaptiveGridStrategy(BaseGridStrategy):
    def __init__(self, ndim = None, nseeds = None, ngrid = None,ref = -3,ref_sig = 0.2,prop_sig = 0.2,nTS_samp = 30,rng=None):
        super().__init__(ndim, nseeds, rng)
        self.lhs = qmc.LatinHypercube(d=self.ndim,rng=self.rng)
        self.ref = ref
        self.ref_sig = ref_sig,
        self.prop_sig = prop_sig
        self.nTS_samp = nTS_samp
        self.n_grid = ngrid

    def loglik(self,ysim, ytrue, err_sig):
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

    def loglik_design(self,model,newX,ytrue,err_sig):
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

        ll = self.loglik(preds['mean'],ytrue,err_sig)
        return ll.squeeze()
    def sample(self,model):
        ref = self.ref
        ref_sig = self.ref_sig,
        prop_sig = self.prop_sig
        nTS_samp = self.nTS_samp
        n_grid = self.n_grid

        # generate LHS
        X = self.lhs.random(n_grid)
        seeds = np.arange(self.nseeds + 1).reshape(-1,1)
        self.Xgrid = make_grid(X,seeds)
        # do importance sample
        w = self.loglik_design(model=model,newX=self.Xgrid,ytrue=ref,err_sig=ref_sig)    
        w = w.squeeze()
        w_prime = w - w.max()
        w_norm = np.exp(w_prime) / np.sum(np.exp(w_prime))
        w_norm = w_norm.squeeze()
        # filter 
        grid_ids = np.random.choice(self.Xgrid.shape[0],size=self.Xgrid.shape[0],p=w_norm, replace=True)
        grid_ids = np.unique(grid_ids)
        w_filtered = w[grid_ids]

        # MH proposals 
        Xgrid_new = []
        for ii in grid_ids:
            if len(Xgrid_new) < self.Xgrid.shape[0]:
                # original candidate
                xs_old = self.Xgrid[ii,:]
                # proposal
                xs_can = self.rng.normal(loc=xs_old[0:self.ndim],scale=prop_sig,size=self.ndim).reshape(1,-1)
                xs_can_mat = make_grid(xs_can,seeds)

                # acceptance prob
                if (xs_can[:,0:self.ndim] < 1).all() and (xs_can[:,0:self.ndim] > 0).all():
                    acc_prob = self.loglik_design(model=model,newX=xs_can_mat,ytrue=ref,err_sig=ref_sig) - w[ii]
                    u = np.log(self.rng.uniform(size=len(seeds)))
                    selected_ids = (u < acc_prob).nonzero()[0]
                    selected_id = sorted(self.rng.choice(selected_ids,size=np.ceil(len(selected_ids)/2).astype(int)))[0]
                    Xgrid_new.append(xs_can_mat[selected_id,:])

        self.Xgrid = np.array(Xgrid_new)
    
        # Thompson samples
        tTS = model.sample(self.Xgrid,size=nTS_samp)
        best_ids = np.argmin(tTS,axis=1)
        best_ids = np.unique(best_ids,axis=0)       
        return self.Xgrid[best_ids]
