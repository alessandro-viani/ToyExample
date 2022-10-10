## Required include

- import scipy
- import pickle
- import copy
- import time
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns
- import scipy.stats as stats


# ToyExample
This repository contains an example of an inverse problem where the aim is to reconstruct the mean of a Gaussian function, given noisy measurements, i.e.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;y(t)=\mathcal{N}_{\xi(t)}(\mu,\sigma)+\varepsilon(t)" title="inverse_problem" />
</p>
<p align="center">

<img src="https://latex.codecogs.com/svg.latex?&space;\varepsilon(t)\sim\mathcal{N}(0,\theta)." title="noise" />
</p>

We assume observations are available on a given set of points and we want to make inference on the mean of the Gaussian distribution

![data_toy](https://raw.githubusercontent.com/alessandro-viani/ToyExample/main/fig/data_toy.png)
> The image shows an example of the data given to the algorithm as blue dots.

#### Prior and Likelihood

We consider a truncated Jeffreys prior on the mean location (uniform in the interval where data are measured), moreover we assume independence between data at different time points, obtaining a simple factorization for the likelihood

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;p^{\theta}(\mathbf{y}\mid\mu)=\prod_{t=1}^Tp^{\theta}(y(t)\mid\mu)" title="likelihood"/>

![plot_confront](https://raw.githubusercontent.com/alessandro-viani/ToyExample/main/fig/plot_confront.png)
> The image shows an example marginals obtained using the proposed method. Notice that the poterior for the noise standard deviation is not obtained as histogram plot due to the fact that the variable is not sampled but the posterior is approximated during the iteration as explained in [^1].

  
[^1]: My reference to be added.

## Particle file

Contains the class Particle, namely one single sample. This class embodies also the functions for the evaluation of the Likelihood and the Prior. This class takes as input some parameters that are passed by the class Posterior explained below.

## Posterior file

Contains the class Posterior, namely a collection of Particles as described above. The class takes as parameter a dictionary _cfg_ containing as instances: the number of particles (n_particles), the estimated noise standard deviation (theta_eff), the interval where data are measured (sourcespace), the measured data (data), the number of bins to use to compute estimates from the histogram of the mean (n_bins), a fixed number of iterations or an adaptive number of iterations (sequence_evolution) and the method one want to use (method).

## Util file

Contains all utilities used in the Classes: Particle and Posterior. Moreover it contains the function forcreating the data utilised for performing the SMC samplers algorithm.
