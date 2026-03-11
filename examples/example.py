from adaptive_ts.example_simulator.odin import generate_ground_truth, run_batch
from scipy.stats.qmc import LatinHypercube
from adaptive_ts.utils import make_grid
import numpy as np
rng = np.random.default_rng(1)
lhs = LatinHypercube(d=2,rng=rng)
seeds = np.arange(5)+1
X = make_grid(lhs.random(5),seeds.reshape(-1,1))
ground_truth = generate_ground_truth()
ytrue = ground_truth['n_SI'].values
batch = run_batch(X,ytrue=ytrue)
Y = batch['Y']
sims = batch['sims']

from adaptive_ts.workflow import Workflow
from adaptive_ts.emulator import crnGPEmulator
from adaptive_ts.grid import AdaptiveGridStrategy, LatinHypercubeStrategy
from adaptive_ts.dataset import Dataset

if __name__ == "__main__":

    adapt = AdaptiveGridStrategy(
            ndim=2,nseeds=5,ngrid=100,
            ref=-2, ref_sig=0.2,prop_sig=0.2, nTS_samp=30, rng = rng,
        )

    workflow = Workflow(
        emulator = crnGPEmulator(
            covtype="Matern5_2"
        ),
        grid_strategy = LatinHypercubeStrategy(
            ndim=2,nseeds=5,rng=rng,nTS_samp=30,ngrid=500
        ),
        budget=200
    )

    nsims = len(Y)
    data = Dataset(X=X,Y=Y)
    
    while nsims < workflow.budget:
        X, Y = data.X, data.Y
        model = workflow.fit_surrogate(X,Y)
        newX = workflow.sample_grid_strategy()
        new_out = run_batch(newX,ytrue=ytrue)
        newY = new_out['Y']
        data.append(X=newX,Y=newY)
        nsims += len(newY)
        print(f'{nsims=}')

    data