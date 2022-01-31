# ToyExample
This repository contains an example of an inverse problem where the aim is to reconstruct the parameters of an unknown number of weighted Gaussian functions, given noisy measurements of their superposition, i.e.

<img src="https://latex.codecogs.com/svg.latex?\large&space;y_k=\sum_{i=1}^da_{i}\mathcal{N}_{\tau_k}(\mu_{i},\sigma_{i})+\varepsilon" title="\large y_k=\sum_{i=1}^da_{i}\mathcal{N}_{\tau_k}(\mu_{i},\sigma_{i})+\varepsilon" />

<img src="https://latex.codecogs.com/svg.latex?\large&space;\varepsilon\sim\mathcal{N}(0,\theta)." title="\large \varepsilon\sim\mathcal{N}(0,\theta)." />

We assume observations are available on a given set of points and we want to make inference on the following parameters: 
- number of Gaussian functions: d
- mean of each Gaussian: \mu
- standard deviation of each Gaussian: \sigma
- amplitude of each Gaussian: a

Since the number of unknowns is itself unknown, the state space is defined as the finite union of fixed dimensional spaces, each one referring to a fixed number of objects. For these class of spaces SMC samplers can be applied using different strategies: in some cases, a different SMC sampler is run for each model, and the normalizing constants are used to compare models; in other cases,  variable-dimension models have been used, so that the dimension of the sample can change from one step to the next one. 

In this example we use the second approach, i.e. we construct Reversible Jump Metropolis Hastings (RJMH) kernels for the particle evolution that allow samples to switch between spaces with different dimensions; in practice, this amounts to having birth and death moves so that a sample with **n** Gaussians can evolve into a sample of **n+1** Gaussians (birth) or **n-1** Gaussians (death).

#### Prior and Likelihood

We assume that all parameters are a priori independent, so that the prior density turns out to be

<img src="https://latex.codecogs.com/svg.latex?\large&space;p(x)=p(d)\prod_{i=1}^{d}p(\mu_i)\prod_{i=1}^{d}p(\sigma_i)\prod_{i=1}^{d}p(a_i)" title="\large p(x)=p(d)\prod_{i=1}^{d}p(\mu_i)\prod_{i=1}^{d}p(\sigma_i)\prod_{i=1}^{d}p(a_i)" />

where:

- <img src="https://latex.codecogs.com/svg.latex?\large&space;x=(d,\mu_{1:d},\sigma_{1:d},a_{1:d})" title="\large x=(d,\mu_{1:d},\sigma_{1:d},a_{1:d})"/>
- <img src="https://latex.codecogs.com/svg.latex?\large&space;p(d)\sim\textit{Poisson}(0.25)" title="\large p(d)\sim\textit{Poisson}(0.25)"/>
- <img src="https://latex.codecogs.com/svg.latex?\large&space;p(\mu_i)\sim\mathcal{U}([-5,5])" title="\large p(\mu_i)\sim\mathcal{U}([-5,5])"/>
- <img src="https://latex.codecogs.com/svg.latex?\large&space;p(\sigma_i)\sim\exp(\mathcal{U}([0.1,10]))" title="\large p(\sigma_i)\sim\exp(\mathcal{U}([0.1,10]))"/>
- <img src="https://latex.codecogs.com/svg.latex?\large&space;p(a_i)\sim\mathcal{N}(1,0.25)" title="\large p(a_i)\sim\mathcal{N}(1,0.25)"/>

Moreover we assume independence between data, obtaining a simple factorization for the likelihood

<img src="https://latex.codecogs.com/svg.latex?\large&space;p^{\theta}(y\mid\textit{x})=\prod_{j=1}^Bp^{\theta}(y_j\mid\textit{d},\mu_{1:d},\sigma_{1:d},a_{1:d})" title="\large p^{\theta}(y\mid\textit{x})=\prod_{j=1}^Bp^{\theta}(y_j\mid\textit{d},\mu_{1:d},\sigma_{1:d},a_{1:d})"/>


## Main file

**This is bold text**
> Text that is a quote
> This site was built using [GitHub Pages](https://pages.github.com/).

![This is an image](https://myoctocat.com/assets/images/base-octocat.svg)

1. First list item
   - First nested list item
     - Second nested list item



## Plot file

Here is a simple footnote[^1].

A footnote can also have multiple lines[^2].  

You can also use words, to fit your writing style more closely[^note].

[^1]: My reference.
[^2]: Every new line should be prefixed with 2 spaces.  
  This allows you to have a footnote with multiple lines.
[^note]:
    Named footnotes will still render with numbers instead of the text but allow easier identification and linking.  
    This footnote also has been made with a different syntax using 4 spaces for new lines.

## SMC file

## Particle file

## Gaussian file
