'''
Workflow class to hold emulator, grid, and seed expansion

Example

workflow = Workflow(
    emulator = crnGPEmulator(),
    grid_strategy = LHSGrid(),
    seed_expansion = SeedExpansion(nsims=10),
    budget = 200
)

X,Y = get_data()
data = Dataset(X,Y)

nsims = len(Y)

while nsims < workflow.budget:
    workflow.emulate(X,Y)
    newX = workflow.grid_strategy()   
    newY = simulator(newX)
    data.append(X,Y)
    nsims += len(newY)

    # check for seed expansion:
    expand_seeds = workflow.seed_expansion(nsims)
    if expand_seeds.condition:
        newX = expand_seeds.expansion
        newY = simulator(newX)
        data.append(X,Y)
        nsims += len(newY)
    
'''

from adaptive_ts.emulator import BaseEmulator
from adaptive_ts.grid import BaseGridStrategy
from adaptive_ts.expansion import BaseSeedExpansion
import numpy as np
from time import time

def _validate_input(name,workflow_input,workflow_expected):
    if not isinstance(workflow_input,workflow_expected):
        raise ValueError(f"{name} is not valid, got {type(workflow_input)} , expected input or inheritance of type {workflow_expected}")

class Workflow:
    def __init__(self,emulator,grid_strategy,seed_expansion_strategy = BaseSeedExpansion(),budget = 200, rng = None):
        # validate emulator
        _validate_input('emulator',emulator,BaseEmulator)
        _validate_input('grid_stategy',grid_strategy,BaseGridStrategy)
        _validate_input('seed_expansion',seed_expansion_strategy,BaseSeedExpansion)
        if rng is None:
            seed = int(time())
            rng = np.random.default_rng(seed)
        if isinstance(rng,int):
            rng = np.random.default_rng(rng)
        elif isinstance(rng,np.random.Generator):
            pass
        else:
            raise ValueError('rng must be either int or numpy.random.Generator')
        self.rng = rng
        # setup rng
        if emulator.rng is None:
            emulator.rng = self.rng
        if grid_strategy.rng is None:
            grid_strategy.rng = self.rng
        if seed_expansion_strategy.rng is None:
            seed_expansion_strategy.rng = self.rng


        self.emulator = emulator
        self.grid_strategy = grid_strategy
        self.seed_expansion_strategy = seed_expansion_strategy
        self.budget = budget
        self.pars = {'budget':budget,
                    'emulator':emulator.__str__(),
                    'grid_strategy':grid_strategy.__str__(),
                    'seed_expansion': seed_expansion_strategy.__str__()
        }
        self.nsims = 0
        # save pars
        for key,value in self.emulator.__dict__.items():
            if isinstance(value,str) or isinstance(value,float) or isinstance(value,int):
                self.pars[key] = value
        for key,value in self.grid_strategy.__dict__.items():
            if isinstance(value,str) or isinstance(value,float) or isinstance(value,int):
                self.pars[key] = value
        return
    
    def fit_surrogate(self,X,Y):
        self.emulator.fit(X,Y)
        return self.emulator
    def sample_grid_strategy(self):
        newX = self.grid_strategy.sample(model=self.emulator)
        self.nsims += len(newX)
        return newX
    
    
    def check_for_expansion(self):

        if self.seed_expansion_strategy.check_for_expansion():
            self.seed_expansion_strategy.expand()
            self.seed_expansion_strategy.nsims = self.nsims
            self.grid_strategy.nseeds = self.seed_expansion_strategy.nseeds
    def sample_from_expansion(self):
        ndim = self.grid_strategy.ndim
        n = self.seed_expansion_strategy.n_expansion
        return self.seed_expansion_strategy.sample(n=n,ndim = ndim)
        