import torch
import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize as P
import numpy as np
from adaptive_ts.emulator import BaseEmulator
import pandas as pd

# botorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
# gpytorch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import IndexKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood



def _check_seed(X):
    '''
    Check that last column of X is integer
    '''
    if isinstance(X,torch.Tensor):
        if not (X[:,-1] == X[:,-1].int()).all(): 
            raise RuntimeError("Last column of X must specify integer random seed")
    elif isinstance(X,np.ndarray):
        if not (X[:,-1] == X[:,-1].astype(int)).all():
                raise RuntimeError("Last column of X must specify integer random seed")
    return

class UnitDiagonal(torch.nn.Module):
    def forward(self,B):
        return F.normalize(B)

class _CustomIndexKernel(IndexKernel):
    def __init__(self, num_tasks, rank = 1, prior = None, var_constraint = None, **kwargs):
        super().__init__(num_tasks, rank, prior, var_constraint, **kwargs)
        P.register_parametrization(self,'covar_factor',UnitDiagonal())
    def _eval_covar_matrix(self):
        #self.covar_factor = F.normalize(self.covar_factor)
        cf = self.covar_factor
        BBT = cf @ cf.transpose(-1, -2)
        diag = BBT.diag()
        out = BBT / torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))
        # omit noise 
        return out + torch.diag_embed(self.var)
        

class _CategoricalSeededGPOnesDiagonal(ExactGP,GPyTorchModel):
    def __init__(self, train_inputs, train_targets, likelihood,covtype="Matern5_2", num_tasks = None,rank = None):
        super().__init__(train_inputs, train_targets, likelihood,)
        _check_seed(train_inputs)
        self.mean_module = ConstantMean()
        self.task_covariance_module = _CustomIndexKernel(num_tasks=num_tasks,rank=rank)
        self.active_dims = np.arange(train_inputs.shape[1] - 1).tolist()
        ard_num_dims = len(self.active_dims)
        # set covariance type
        if covtype == 'Matern5_2':
            covar = MaternKernel(nu=2.5,active_dims=self.active_dims,ard_num_dims=ard_num_dims)
        elif covtype == 'Gaussian':
            covar = RBFKernel(active_dims=self.active_dims,ard_num_dims=ard_num_dims)
        elif isinstance(covtype,gpytorch.kernels.Kernel):
            covar = covtype
        self.covar_module = ScaleKernel(covar)

    def forward(self, x):
        if not self.training:
            foo = 1

        X_continuous = x[:,self.active_dims]
        X_cat = (x[:,-1] - 1).int() # categoricals must map to {0, ..., (N - 1)}
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(X_continuous) * self.task_covariance_module(X_cat)
        return MultivariateNormal(mean_x, covar_x)

class SeedlingGP(BaseEmulator):
    def __init__(self,likelihood = GaussianLikelihood(),covtype = "Matern5_2",num_tasks = None, rank = None,variance_cutoff = 0.95):
        super().__init__()
        self.likelihood = likelihood
        self.num_tasks = num_tasks
        self.rank = rank
        self.covtype = covtype
        self.variance_cutoff = variance_cutoff
    def __str__(self):
        return 'SeedlingGP'
    def fit(self,X,Y):
        self.model = None
        if isinstance(X,np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(Y,np.ndarray):
            Y = torch.from_numpy(Y)
        # set rank based on X
        if self.rank == "dynamic":
            # make a dataframe based on X
            cols = [f'x{i}' for i in range(X.shape[1]-1)]
            df = pd.DataFrame(X,columns=cols + ['seed'])
            df['Y'] = Y
            corr_matrix = df.pivot_table(index=cols,columns=['seed'],values='Y').reset_index(drop=True).interpolate(method='cubic').corr()
            try:
                U,s,Vh = np.linalg.svd(corr_matrix)
                s_squared = s**2
                total_variance = s_squared.sum()
                cumulative_explained_variance = (s_squared / total_variance).cumsum()
                
                cutoff = np.max([cumulative_explained_variance.min(),self.variance_cutoff])
                self.rank = (cumulative_explained_variance <= cutoff).nonzero()[0][-1] + 1
                print(f"Dynamic rank is {self.rank}")
            except np.linalg.LinAlgError as e:
                print('SVD did not converge, setting rank to 1')
                self.rank = 1


        model = _CategoricalSeededGPOnesDiagonal(train_inputs = X, train_targets = Y,
                                                 likelihood = self.likelihood,
                                                 covtype = self.covtype,
                                                 num_tasks = self.num_tasks, rank = self.rank)
        model.train()
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood,model=model)
        fit_gpytorch_mll(mll)
        
        model.eval()
        self.model = model

    def predict(self,X,return_cov = False):
        with torch.no_grad():
            if isinstance(X,np.ndarray):
                X = torch.from_numpy(X)
            post = self.model.posterior(X, observation_noise=True)
            preds = {'mean': post.mean.detach().numpy().squeeze(),'cov':None}
            if return_cov:
                preds['cov'] = post.covariance_matrix.detach().numpy()
        return preds

    def sample(self,X,size=1):
        with torch.no_grad():
            if isinstance(X,np.ndarray):
                X = torch.from_numpy(X)
            preds = self.model.posterior(X=X,observation_noise=True)

        return preds.sample(sample_shape=torch.Size([size])).squeeze()