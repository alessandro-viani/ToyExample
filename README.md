# ToyExample
This repository contains an example of an inverse problem where the aim is to reconstruct the parameters of an unknown number of weighted Gaussian functions, given noisy measurements of their superposition, i.e.

<img src="https://latex.codecogs.com/svg.latex?&space;y_k=\sum_{i=1}^da_{i}\mathcal{N}_{\tau_k}(\mu_{i},\sigma_{i})+\varepsilon" title="y_k=\sum_{i=1}^da_{i}\mathcal{N}_{\tau_k}(\mu_{i},\sigma_{i})+\varepsilon" />

<img src="https://latex.codecogs.com/svg.latex?&space;\varepsilon\sim\mathcal{N}(0,\theta)." title="\varepsilon\sim\mathcal{N}(0,\theta)." />

We assume observations are available on a given set of points and we want to make inference on the following parameters: 
- number of Gaussian functions: _d_
- mean of each Gaussian: μ
- standard deviation of each Gaussian: σ
- amplitude of each Gaussian: _a_

![data](https://user-images.githubusercontent.com/57596360/152046064-384f1238-20b2-49eb-9f4e-cc01c37be279.png)
> The image shows an example of data used as blue dots and the forward model as a gray line.

Since the number of unknowns is itself unknown, the state space is defined as the finite union of fixed dimensional spaces, each one referring to a fixed number of objects. For these class of spaces SMC samplers can be applied using different strategies: in some cases, a different SMC sampler is run for each model, and the normalizing constants are used to compare models; in other cases,  variable-dimension models have been used, so that the dimension of the sample can change from one step to the next one. 

In this example we use the second approach, i.e. we construct Reversible Jump Metropolis Hastings (RJMH) kernels for the particle evolution that allow samples to switch between spaces with different dimensions; in practice, this amounts to having birth and death moves so that a sample with **n** Gaussians can evolve into a sample of **n+1** Gaussians (birth) or **n-1** Gaussians (death).

#### Prior and Likelihood

We assume that all parameters are a priori independent, so that the prior density turns out to be

<img src="https://latex.codecogs.com/svg.latex?&space;p(x)=p(d)\prod_{i=1}^{d}p(\mu_i)\prod_{i=1}^{d}p(\sigma_i)\prod_{i=1}^{d}p(a_i)" title="p(x)=p(d)\prod_{i=1}^{d}p(\mu_i)\prod_{i=1}^{d}p(\sigma_i)\prod_{i=1}^{d}p(a_i)" />

where:

- <img src="https://latex.codecogs.com/svg.latex?&space;x=(d,\mu_{1:d},\sigma_{1:d},a_{1:d})" title="x=(d,\mu_{1:d},\sigma_{1:d},a_{1:d})"/>
- <img src="https://latex.codecogs.com/svg.latex?&space;p(d)\sim\textit{Poisson}(0.25)" title="p(d)\sim\textit{Poisson}(0.25)"/>
- <img src="https://latex.codecogs.com/svg.latex?&space;p(\mu_i)\sim\mathcal{U}([-5,5])" title="p(\mu_i)\sim\mathcal{U}([-5,5])"/>
- <img src="https://latex.codecogs.com/svg.latex?&space;p(\sigma_i)\sim\exp(\mathcal{U}([0.1,10]))" title="p(\sigma_i)\sim\exp(\mathcal{U}([0.1,10]))"/>
- <img src="https://latex.codecogs.com/svg.latex?&space;p(a_i)\sim\mathcal{N}(1,0.25)" title="p(a_i)\sim\mathcal{N}(1,0.25)"/>

Moreover we assume independence between data, obtaining a simple factorization for the likelihood

<img src="https://latex.codecogs.com/svg.latex?&space;p^{\theta}(y\mid\textit{x})=\prod_{j=1}^Bp^{\theta}(y_j\mid\textit{d},\mu_{1:d},\sigma_{1:d},a_{1:d})" title="p^{\theta}(y\mid\textit{x})=\prod_{j=1}^Bp^{\theta}(y_j\mid\textit{d},\mu_{1:d},\sigma_{1:d},a_{1:d})"/>


![plot_marginals](https://user-images.githubusercontent.com/57596360/152060533-a6278473-1fbb-430c-8c1e-89345d9d841c.png)
> The image shows an example marginals obtained using the proposed method. Notice that the poterior for the noise standard deviation is not obtained as histogram plot due to the fact that the variable is not sampled but the posterior is approximated during the iteration as explained in [^1].

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

- num_evolution: if _None_ the number is one of the parameters to be estimated otherwise if _int_ it represents the fixed number of gaussian for each particle
- mean_evolution: if _True_ the mean is one of the parameters to be estimated otherwise if _False_ the mean is fixed as the true mean value [works with one gaussian only]
- std_evolution: if _True_ the standard deviation is one of the parameters to be estimated otherwise if _False_ the standard deviation is fixed as the true standard deviation value [works with one gaussian only]
- amp_evolution: if _True_ the amplitude is one of the parameters to be estimated otherwise if _False_ the amplitude is fixed as the true amplitude value [works with one gaussian only]
- prop_method: if _True_ the noise standard deviation is one of the parameters to be estimated otherwise if _False_ the noise standard deviation is fixed as the estimated noise standard deviation value and at the very last iteration recycle scheme and noise posterior are estimated using the proposed method
- sequence_evolution: if _None_ the number of iteration is adaptively chosen otherwise if _int_ it represents the fixed number of iterations for SMC samplers
- mh_evolution: if _True_ the number of Metropolis hastings step is adaptively chosen otherwise if _False_ the Metropolis Hastings step at each iteration is one
- sourcespace: _x_ axes values for the sampled data
- data: _y_ axes noisy values measured
- max_exp: maximum exponent used for the likelihood function
- n_particles: the number of particles for performing SMC samplers
- max_num: maximum number of Gaussian in each Particle
- noise_std_eff: the estimated noise standard deviation
- prior_num: mean for the Poisson prior on the number of Gaussian
- prior_m: interval extremes for the Uniform prior on Gaussian mean
- prior_s: interval extremes for the Log-Uniform prior on Gaussian standard deviation
- prior_a: mean and variance Normal prior on Gaussian amplitude
- prior_n: shape and scale parameters for the Gamma prior on the noise standard deviation

## Util file

Contains all utilities used in the Classes: Particle and Posterior

## PerformanceAnalysis

This file contains all needed for performing analysis on the proposed method against the classical one. The parameters to choos at first are:

- n_file: number of simulations one want to analise
- noise_range: range for the uniform distributed noise standard deviation for creating the data
- n_data: number of data to use in each simulation
- n_particles: number of particles to use in each run for both proposed and classical method
