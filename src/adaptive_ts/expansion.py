import numpy as np
class BaseSeedExpansion:
    def __init__(self, nseeds: int = None, rng: int | np.random.Generator  = None):
        self.nseeds = nseeds
        self.nsims = 0
        # validate random number generator
        if rng is None or isinstance(rng,np.random.Generator):
            pass
        elif isinstance(rng,int):
            rng = np.random.default_rng(rng)
        else:
            raise ValueError('rng must be either int or numpy.random.Generator')
        self.rng = rng
        
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