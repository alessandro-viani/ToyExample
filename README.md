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
This repository contains an example of an inverse problem where the aim is to reconstruct the parameters of an unknown number of weighted Gaussian functions, given noisy measurements of their superposition, i.e.

<p align="center">

<img src="https://latex.codecogs.com/svg.latex?&space;y(t)=\mathcal{N}_{\xi(t)}(\mu,\sigma)+\varepsilon(t)" title="data" />
</p>

<p align="center">

<img src="https://latex.codecogs.com/svg.latex?&space;\varepsilon(t)\sim\mathcal{N}(0,\theta)." title="noise" />
</p>

We assume observations are available on a given set of points and we want to make inference on the mean of the Gaussian distribution

#### Prior and Likelihood

We consider a uniform prior on the mean location in the interval where data are measured, moreover we assume independence between data at different time points, obtaining a simple factorization for the likelihood

<p align="center">
  
<img src="https://latex.codecogs.com/svg.latex?&space;p^{\theta}(\mathbf{y}\mid\textit{x})=\prod_{t=1}^Tp^{\theta}(y(t)\mid\mu)" title="likelihood"/>

![plot_marginals](https://user-images.githubusercontent.com/57596360/152060533-a6278473-1fbb-430c-8c1e-89345d9d841c.png)
> The image shows an example marginals obtained using the proposed method. Notice that the poterior for the noise standard deviation is not obtained as histogram plot due to the fact that the variable is not sampled but the posterior is approximated during the iteration as explained in [^1].
<\p>
  
[^1]: My reference to be added.

## Gaussian file

Contains the class Gaussian, nothing but the parameters of interest of the Gaussian to be reconstructed: apmplitude, mean and variance.

## Particle file

Contains the class Particle, namely one single sample containing some Gaussian. This class embodies also the functions for the evaluation of the Likelihood and the prior as well as the prior functions definitions. This class takes as input some parameters that are passed by the class SMC therefore we explain them below.

## SMC file

Contains the class Posterior, namely a collection of Particles as described above. The class takes as parameters some boolean that tells which variables needs to be evolved as well as the parameters for the prior passed to the class Particle.

The class contains all functions needed for performing SMC samplers and, if required, also the proposed method where all particles are recycled and the posterior for the parameter _noise standard deviation_ is approximated without any additional computational cost.

## Main file

The main file contains all nedeed for running the code setting the parameters of the class SMC, namely:

- prop_method: if _True_ the noise standard deviation is one of the parameters to be estimated otherwise if _False_ the noise standard deviation is fixed as the estimated noise standard deviation value and at the very last iteration recycle scheme and noise posterior are estimated using the proposed method
- sequence_evolution: if _None_ the number of iteration is adaptively chosen otherwise if _int_ it represents the fixed number of iterations for SMC samplers
- sourcespace: _x_ axes values for the sampled data
- data: _y_ axes noisy values measured
- max_exp: maximum exponent used for the likelihood function
- n_particles: the number of particles for performing SMC samplers
- theta_eff: the estimated noise standard deviation
- prior_mean: interval extremes for the Uniform prior on Gaussian mean
- prior_theta: shape and scale parameters for the Gamma prior on the noise standard deviation

## Util file

Contains all utilities used in the Classes: Particle and Posterior. Moreover it contains the function forcreating the data utilised for performing the SMC samplers algorithm.
  
The creation of the data is demanded to the function *creation_data* that takes as input the mean, standard deviation and amplitude of each gaussian as an array as long as other parameters.
  
Example: for creating a data coming form the superpostion of two Gaussians with mean -2 and 2, standard deviation 1 and 0.5 and amplitude 0.7 and 0.3 in the range [-5,5] one just use the function as follows:
  
> creation_data(n_data=100, mean=[-2,2], std=[1, 0.5], amp=[0.7,0.3], min_max_sour=[-5,5])

## Analysis

This file contains all needed for performing analysis on the proposed method against the classical one. The parameters to choos at first are:

- n_file: number of simulations one want to analise
- noise_range: range for the uniform distributed noise standard deviation for creating the data
- n_data: number of data to use in each simulation
- n_particles: number of particles to use in each run for both proposed and classical method

> warning: use a high enought number of particles and data in this section, at least _n_data>50_ and _n_particles>50_ and a noise standard deviation range within _[0.01, 0.5]_
