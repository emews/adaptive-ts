# adaptive-ts
Adaptive emulation and Thompson sampling in Python

The `adaptive_ts` package is a Python library to faciliate adaptive sampling from stochastic simulations using emulators.

The package is designed for calibrating stochastic simulators where:
*   Extensive simulation is infeasible or prohibitively expensive (i.e. HPC simulators taking minutes or hours to run)
*   The simulator output can be mapped to some notion of ''observed'' (often real-world) data
*   There structure within the stochastic component, given by a random seed

## Installation
*   `adaptive_ts` may be install via pip and github (pypi release coming soon)

```{bash}
pip install git+https://github.com/emews/adaptive-ts.git
```


## Examples

A tutorial of the `adaptive_ts` package may be found at https://github.com/emews/wsc2026-TOO-tutorial
