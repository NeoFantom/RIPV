# RIPV

This repo is dedicated to the paper introducing *Robustness-Invariant Pulse Variation* algorithm, containning codes developed by [Huiqi Xue](https://github.com/NeoFantom).

The original paper can be found on [arXiv:2412.19473](https://arxiv.org/abs/2412.19473). Publication is in preparation.

## ALgorithm introduction

The RIPV algorithm, as the name suggests, varies quantum control pulses without changing its robustness. Essentially, it traverses the level set of the robustness landscape (robustness function).

RIPV is an application of the Gradient Orthogonal Variation (GOV) algorithm, where, instead of going along the gradient to maximize/minimize a function, you move in a direction perpendicular to the gradient to keep this function constant. The GOV algorithm is rather general, irrelevant to our quantum control background. Given $n$ free variables, GOV can keep at most $n-1$ constraints constant at the initial value. This algorithm is already used in Computer Science (CS) but we have not found a commonly accepted name (e.g. null space method could mean this algorithm, depending on context). Hence we named it GOV to emphasize its orthogonality in comparison to Gradient Descent (GD).

## File structure

3 types of files are provided: beginning pulses (.csv), library functions (.py), Python scripts for numerical experiments (.ipynb):

### Beginning pulses
RIPV does not contain any randomness and therefore we only provide the beginning pulses: $R_x(\pi)$ (1st-order robust to arbitrary single qubit noise) $R_x(2\pi)$ (2nd-order robust to $\sigma_z$), which are actually from [Hai et al.](https://arxiv.org/abs/2210.14521). More code and data can be found [here](https://github.com/QDynamics/RobustControl). Other pulses obtained through variation can be produced by running the scripts.

### Library functions

These files contain the [control/dynamical model](https://github.com/NeoFantom/RIPV/blob/master/RIPV_control_models.py), [pulse parametrization model](https://github.com/NeoFantom/RIPV/blob/master/RIPV_pulseModel.py), [propagator and integral calculation](https://github.com/NeoFantom/RIPV/blob/master/RIPV_import.py), and [GOV implementation](https://github.com/NeoFantom/RIPV/blob/master/RIPV_core.py) used by RIPV.

### Python notebooks

These are used by the author to generate robust control pulses for gate families.

- [Experiments](https://github.com/NeoFantom/RIPV/blob/master/RIPV-Autodiff-Experiments.ipynb): generates robust $R_x(\theta)$ gates against $\sigma_z$ noise
- [3Noise](https://github.com/NeoFantom/RIPV/blob/master/RIPV-Autodiff-3Noise.ipynb) generates robust $R_x(\theta)$ gates against $\sigma_x$, $\sigma_y$ and $\sigma_z$ noise
- [Plotting](https://github.com/NeoFantom/RIPV/blob/master/RIPV-Plotting.ipynb) is used to generate colorful plottings, dedicated for continuously-varying pulse-family visuallization.

_Will be improved soon_
