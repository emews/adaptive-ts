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

def _validate_input(name,workflow_input,workflow_expected):
    if not isinstance(workflow_input,workflow_expected):
        raise ValueError(f"{name} is not valid, got {type(workflow_input)} , expected input or inheritance of type {workflow_expected}")

class Workflow:
    def __init__(self,emulator,grid_strategy,seed_expansion_strategy = None,budget = 200):
        # validate emulator
        _validate_input('emulator',emulator,BaseEmulator)
        _validate_input('grid_stategy',grid_strategy,BaseGridStrategy)

        self.emulator = emulator
        self.grid_strategy = grid_strategy
        self.budget = budget
        return
    
    def fit_surrogate(self,X,Y):
        self.emulator.fit(X,Y)
        return self.emulator
    def sample_grid_strategy(self):
        newX = self.grid_strategy.sample(model=self.emulator)
        return newX