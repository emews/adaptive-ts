from time import time
import numpy as np

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
    def __str__(self):
        return 
    def sample(self):
        raise NotImplementedError()

class FixedGridStrategy(BaseGridStrategy):
    def __init__(self,ndim: int = None,nseeds:int = None, rng = None):
        super().__init__(ndim=ndim, nseeds=nseeds, rng=rng)
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
        seeds = (np.arange(self.nseeds)+1).reshape(-1,1)
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

        self.Xgrid = np.array(Xgrid_new)
    
        # Thompson samples
        tTS = model.sample(self.Xgrid,size=nTS_samp)
        best_ids = np.argmin(tTS,axis=1)
        best_ids = np.unique(best_ids,axis=0)       
        return self.Xgrid[best_ids]


class NormalizingFlowStrategy(BaseGridStrategy):
    def __init__(self, ndim = None, nseeds = None, ngrid = None,ref = -3,ref_sig = 0.2, nTS_samp = 30,rng=None):
        super().__init__(ndim, nseeds, rng)
        self.lhs = qmc.LatinHypercube(d=self.ndim,rng=self.rng)
        self.ref = ref
        self.ref_sig = ref_sig,
        self.nTS_samp = nTS_samp
        self.n_grid = ngrid
    
    def __str__(self):
        super().__str__()
        return 'NormalizingFlows'
    def _train_flow(self,df,nsamples,seeds):
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
        seed = self.rng.integers(low=1,high=100_000)
        torch.manual_seed(seed)
        # Load samples
        weights = df["weight"].values
        r_data = pd.Categorical(df["r"].astype(int),categories=seeds)
        x_columns = [c for c in df.columns if c not in ["r", "weight"]]

        # Logit-transform x (maps [0,1] -> R)
        x_data = df[x_columns].to_numpy()
        x_data = logit(x_data)

        # One-hot encode r
        r_onehot = pd.get_dummies(r_data).to_numpy()

        # Combine features
        x_cond = np.hstack([x_data, r_onehot])
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
        flow = Flow(transform, base_dist)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            log_prob = flow.log_prob(inputs=x_tensor)
            loss = -(w_tensor * log_prob).sum()
            loss.backward()
            optimizer.step()

        # Sample from the flow
        with torch.no_grad():
            samples = flow.sample(nsamples).numpy()

        # Split x and r back
        n_x = len(x_columns)
        x_samples = inv_logit(samples[:, :n_x])  # map back to [0,1]

        r_cols = [col for col in pd.get_dummies(r_data).columns]
        r_indices = samples[:, n_x:].argmax(axis=1)
        r_decoded = [r_cols[i] for i in r_indices]

        # Save output
        out_df = pd.DataFrame(x_samples, columns=x_columns)
        out_df["r"] = r_decoded

        del flow
        del optimizer
        del loss
        return out_df
    def sample(self,model):
        ref = self.ref
        ref_sig = self.ref_sig,
        nTS_samp = self.nTS_samp
        n_grid = self.n_grid

        # generate LHS
        X = self.lhs.random(n_grid)
        seeds = (np.arange(self.nseeds)+1).reshape(-1,1)
        self.Xgrid = make_grid(X,seeds)
        # do importance sample
        w = loglik_design(model=model,newX=self.Xgrid,ytrue=ref,err_sig=ref_sig)    
        w = w.squeeze()
        w_prime = w - w.max()
        w_norm = np.exp(w_prime) / np.sum(np.exp(w_prime))
        w_norm = w_norm.squeeze()

        # filter and sample 
        grid_ids = self.rng.choice(self.Xgrid.shape[0],size=self.Xgrid.shape[0],p=w_norm, replace=True)
        grid_ids = np.unique(grid_ids)
        cols = [f'x{i+1}' for i in range(self.ndim)]
        cols.append('r')
        grid_df = pd.DataFrame(self.Xgrid[grid_ids,:],columns=cols)
        w_filtered = w[grid_ids]

        w_prime = w_filtered - w_filtered.max()
        w_norm  = np.exp(w_prime) / np.sum(np.exp(w_prime))
        grid_df['weight'] = w_norm

        generated_df = self._train_flow(grid_df,nsamples=1_000,seeds = seeds.squeeze())
        self.Xgrid = generated_df.values


        tTS = model.sample(self.Xgrid,size=nTS_samp)
        best_ids = np.argmin(tTS,axis=1)
        best_ids = np.unique(best_ids,axis=0)
        return self.Xgrid[best_ids,:]
