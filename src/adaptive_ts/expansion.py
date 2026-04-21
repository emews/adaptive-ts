import numpy as np
from scipy.stats.qmc import LatinHypercube
from adaptive_ts.utils import make_grid
class BaseSeedExpansion:
    def __init__(self, nseeds: int = None,n_expansion: int = None,rng: int | np.random.Generator  = None):
        self.nseeds = nseeds
        self.nsims = 0
        self.n_expansion = n_expansion
        # validate random number generator
        if rng is None or isinstance(rng,np.random.Generator):
            pass
        elif isinstance(rng,int):
            rng = np.random.default_rng(rng)
        else:
            raise ValueError('rng must be either int or numpy.random.Generator')
        self.rng = rng
    def sample(self,n:int = None,ndim: int = None):
        '''
        Basic implementation of an exploratory LHS and the maximum seed

        Usually called after self.expand()
        '''
        if n is None:
            n = self.n_expansion
        if n > 0:
            lhs = LatinHypercube(d = ndim, rng = self.rng)  
            max_seed = np.array(self.nseeds).reshape(1,1)
            X = lhs.random(n=n)
            return make_grid(X,max_seed)

    def expand(self):
        self.nseeds += 1
        seeds = (np.arange(1,self.nseeds) + 1).astype(int)
        return seeds
            
            
class SeedExpansionBySims(BaseSeedExpansion):
    def __init__(self, nseeds = None, nsims_expand:int = None, rng = None):
        super().__init__(nseeds, rng)
        self.nsims_expand = nsims_expand
    def __str__(self):
        return "SeedExpansionBySims"
    
    def check_for_expansion(self):
        return (self.nsims % self.nsims_expand) == 0

class SeedExpansionByProb(BaseSeedExpansion):
    def __init__(self,nseeds,prob_expand = None, rng = None):
        super().__init__(nseeds = nseeds, rng = rng)
        self.prob_expand = prob_expand
    def __str__(self):
        return "SeedExpansionByProb"
    def check_for_expansion(self):
        return self.rng.uniform() < self.prob_expand