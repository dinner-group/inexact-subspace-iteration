# Inexact subspace iteration

This repository contains code to run the inexact iterative numerical algebra schemes for spectral estimation and forecasting from [https://arxiv.org/abs/2303.12534](https://arxiv.org/abs/2303.12534).

## Requirements
 - `python=3.8|3.9`
 - `jax`
 - `optax` for optimizers
 - `numpy`
 - `scipy`
 - `numba` to for numerical acceleration
 
See the [Jax repository](https://github.com/google/jax#installation) for more detailed installation instructions on GPUs.
 
## Instructions
The Jupyter notebook [VPM_Muller_Brown.ipynb](VPM_Muller_Brown.ipynb) contains template code to simulate the Müller-Brown potential (as described in the paper)
and solve the eigenproblem with subspace iteration and forecast problems (MFPT and committor).
The neural-network training will be most efficient with a GPU.
