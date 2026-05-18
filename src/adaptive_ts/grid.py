from time import time
import numpy as np
from copy import deepcopy
from scipy.stats import qmc
from adaptive_ts.utils import make_grid, loglik, loglik_design, logit, inv_logit
from scipy.stats import norm
import pandas as pd

# normalizing flows
import torch
torch.set_default_dtype(torch.float64)
from nflows.distributions import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform


class BaseGridStrategy:
    r'''
    Base class for grid strategy. Defines an __init__, __str__, and sample method 
    '''
    def __init__(self, ndim:int = None, nseeds:int = None, rng: np.random.Generator | int | None = None):
        r'''
        Minimal parameters needed for a grid strategy.

        Parameters
        ----------
        ndim: int 
            number of continuous parameters
        nseeds: int
            number of random seeds
        rng: np.random.Generator
            numpy random number generator or integer to seed the generator. If None, the current time is used as the seed

        '''
        self.ndim = ndim
        self.nseeds = nseeds

        # validate random number generator
        if rng is None or isinstance(rng,np.random.Generator):
            pass
        elif isinstance(rng,int):
            rng = np.random.default_rng(rng)
        else:
            raise ValueError('rng must be either int or numpy.random.Generator')
        self.rng = rng
        # set Xgrid
        self.Xgrid = None
    def __str__(self):
        return 
    def sample(self):
        raise NotImplementedError()
    def copy(self):
        '''
        Return a copy of the model
        '''
        newgrid = deepcopy(self)
        return newgrid

class FixedGridStrategy(BaseGridStrategy):
    r'''
    Fixed grid (non-adaptive) strategy. 
    '''
    def __init__(self,ndim: int = None,nseeds:int = None, rng = None,Xgrid = None):
        super().__init__(ndim=ndim, nseeds=nseeds, rng=rng)
        if Xgrid is not None:
            self.Xgrid = Xgrid
    def __str__(self):
        super().__str__()
        return 'FixedGrid'
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
    def __str__(self):
        super().__str__()
        return 'LHSGrid'
    def sample(self,model):
        X = self.lhs.random(self.ngrid)
        if self.nseeds is not None:
            seeds = (np.arange(self.nseeds)+1).reshape(-1,1)
        else:
            seeds = None
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
    def __str__(self):
        super().__str__()
        return 'AdaptiveGrid'
    
    def sample(self,model):
        ref = self.ref
        ref_sig = self.ref_sig,
        prop_sig = self.prop_sig
        nTS_samp = self.nTS_samp
        n_grid = self.n_grid

        # generate LHS
        X = self.lhs.random(n_grid)
        if self.nseeds is not None:
            seeds = (np.arange(self.nseeds)+1).reshape(-1,1)
        else:
            seeds = None
        self.Xgrid = make_grid(X,seeds)
        # do importance sample
        w = loglik_design(model=model,newX=self.Xgrid,ytrue=ref,err_sig=ref_sig)    
        w = w.squeeze()
        w_prime = w - w.max()
        w_norm = np.exp(w_prime) / np.sum(np.exp(w_prime))
        w_norm = w_norm.squeeze()
        # filter 
        grid_ids = self.rng.choice(self.Xgrid.shape[0],size=self.Xgrid.shape[0],p=w_norm, replace=True)
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
                    acc_prob = loglik_design(model=model,newX=xs_can_mat,ytrue=ref,err_sig=ref_sig) - w[ii]
                    u = np.log(self.rng.uniform(size=len(seeds)))
                    selected_ids = (u < acc_prob).nonzero()[0]
                    if len(selected_ids)>0:
                        selected_id = sorted(self.rng.choice(selected_ids,size=np.ceil(len(selected_ids)/2).astype(int)))[0]
                        Xgrid_new.append(xs_can_mat[selected_id,:])
        if len(Xgrid_new) == 0:
            Xgrid_new = self.Xgrid[self.rng.choice(len(self.Xgrid)),:].reshape(-1,self.Xgrid.shape[1])
        self.Xgrid = np.array(Xgrid_new)
    
        # Thompson samples
        tTS = model.sample(self.Xgrid,size=nTS_samp)
        best_ids = np.argmin(tTS,axis=1)
        best_ids = np.unique(best_ids,axis=0)       
        return self.Xgrid[best_ids]


class NormalizingFlowStrategy(BaseGridStrategy):
    def __init__(self, ndim = None, nseeds = None, ngrid = None,persist = False,n_flow_steps = 100,ref = -3,ref_sig = 0.2, nTS_samp = 30,rng=None):
        super().__init__(ndim, nseeds, rng)
        self.lhs = qmc.LatinHypercube(d=self.ndim,rng=self.rng)
        self.ref = ref
        self.ref_sig = ref_sig,
        self.nTS_samp = nTS_samp
        self.n_grid = ngrid
        self.persist = persist
        self.n_flow_steps = n_flow_steps
        self.flow = None
        self.optimizer = None
    def __str__(self):
        super().__str__()
        return 'NormalizingFlows'
    def _train_flow(self,df,weights,nsamples,seeds):
        r'''
        Train normalizing flow and sample from it

        Parameters
        ----------
        df: pd.DataFrame
            dataframe of observations. must contain calibration parameters, weights, and seeds
        nsamples: int
            number of samples from normalizing flow
        
        Returns
        -------
        out_df: pd.DataFrame
            dataframe with sampled outputs (and seeds)

        '''
        rand_seed = self.rng.integers(low=1,high=100_000)
        torch.manual_seed(rand_seed)
        # Load samples
        x_columns = [c for c in df.columns if c not in ["r", "weight"]]
        x_data = df[x_columns].to_numpy()
        # Logit-transform x (maps [0,1] -> R)
        x_data = logit(x_data)
        if seeds is not None:
            # One-hot encode r
            seeds = seeds.squeeze()
            r_data = pd.Categorical(df["r"].astype(int),categories=seeds)
            r_onehot = pd.get_dummies(r_data).to_numpy()
            # Combine features
            x_cond = np.hstack([x_data, r_onehot])
        else:
            # ignore seed and just use continuous pars
            x_cond = x_data.copy()
        
        x_tensor = torch.tensor(x_cond, dtype=torch.float64)
        w_tensor = torch.tensor(weights / np.sum(weights), dtype=torch.float64)

        # Define flow
        n_features = x_tensor.shape[1]
        transform = CompositeTransform([
            MaskedAffineAutoregressiveTransform(
                features=n_features,
                hidden_features=64,
                num_blocks=2
            ) for _ in range(3)
        ])
        base_dist = StandardNormal([n_features])
        if self.flow is None:
            self.flow = Flow(transform, base_dist)
            self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=1e-3)

        # Train
        for epoch in range(self.n_flow_steps):
            self.optimizer.zero_grad()
            log_prob = self.flow.log_prob(inputs=x_tensor)
            loss = -(w_tensor * log_prob).sum()
            loss.backward()
            self.optimizer.step()

        # Sample from the flow
        with torch.no_grad():
            samples = self.flow.sample(nsamples).numpy()

        
        # Split x and r back
        n_x = len(x_columns)
        x_samples = inv_logit(samples[:, :x_data.shape[1]])  # map back to [0,1]

        
        # Save output
        out_df = pd.DataFrame(x_samples, columns=x_columns)
        if seeds is not None:
            r_cols = [col for col in pd.get_dummies(r_data).columns]
            r_indices = samples[:, n_x:].argmax(axis=1)
            r_decoded = [r_cols[i] for i in r_indices]
            out_df["r"] = r_decoded
        if not self.persist:
            self.flow = None
            self.optimizer = None
            del loss
        return out_df
    def sample(self,model):
        ref = self.ref
        ref_sig = self.ref_sig,
        nTS_samp = self.nTS_samp
        n_grid = self.n_grid

        # generate LHS
        X = self.lhs.random(n_grid)
        if self.nseeds is not None:
            seeds = (np.arange(self.nseeds)+1).reshape(-1,1)
        else:
            seeds = None
        self.Xgrid = make_grid(X,seeds)
        # do importance sample
        w = loglik_design(model=model,newX=self.Xgrid,ytrue=ref,err_sig=ref_sig)    
        if len(w.shape) == 1:
            w = w.reshape(-1,1)
        w_prime = w - w.max(axis=0)
        w_norm = np.exp(w) / np.sum(np.exp(w),axis=0,keepdims=True)
        w_norm = w_norm.sum(axis=1)
        w_norm /= w_norm.sum()
        w_norm = w_norm.squeeze()

        # filter and sample 
        grid_ids = self.rng.choice(self.Xgrid.shape[0],size=self.Xgrid.shape[0],p=w_norm, replace=True)
        grid_ids = np.unique(grid_ids)
        cols = [f'x{i+1}' for i in range(self.ndim)]
        if seeds is not None:
            cols.append('r')
        grid_df = pd.DataFrame(self.Xgrid[grid_ids,:],columns=cols)
        w_filtered = w[grid_ids,:]

        w_prime = w_filtered - w_filtered.max()
        w_norm  = np.exp(w_prime) / np.sum(np.exp(w_prime),axis=0,keepdims=True)
        w_norm = w_norm.sum(axis=1)
        w_norm /= w_norm.sum()
        w_norm = w_norm.squeeze()
        generated_df = self._train_flow(grid_df,w_norm,nsamples=n_grid,seeds = seeds)
        self.Xgrid = generated_df.values


        #tTS = model.sample(self.Xgrid,size=nTS_samp)
        #best_ids = np.argmin(tTS,axis=1)
        #best_ids = np.unique(best_ids,axis=0)
        return self.Xgrid


    
    
    
    


