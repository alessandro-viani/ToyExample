"""
Created on May 2022

@author: Alessandro Viani
"""
import pickle

from Posterior import Posterior
from Util import creation_data

# parameters
theta_true = 0.1
n_data = 20
n_particles = 50
verbose = True
ess_min = 0.9
ess_max = 0.99
folder_name = 'sol/'

prior_mean = [-5, 5]
prior_theta = [2, 4]

saturation = 0.7
linewidth = 1
width = 0.7
dpi = 50
plot_show = True
size_ticks = 22
n_bins = 50

sourcespace, data = creation_data(n_data=n_data, theta=theta_true)
with open('data/data.pkl', 'wb') as f:
    pickle.dump([sourcespace, data, theta_true], f)

# %% RUN PROPOSED
post = Posterior(prop_method=True,
                 sourcespace=sourcespace,
                 data=data,
                 n_particles=n_particles,
                 theta_eff=theta_true / 2,
                 prior_mean=prior_mean,
                 prior_theta=prior_theta,
                 ess_min=ess_min,
                 ess_max=ess_max,
                 verbose=verbose)

post = post.perform_smc()

with open(f'{folder_name}/sol.pkl', 'wb') as f:
    pickle.dump(post, f)

# %% RUN CLASSICAL
post_clas = Posterior(prop_method=False,
                      sourcespace=sourcespace, data=data,
                      n_particles=n_particles, theta_eff=theta_true / 2,
                      prior_mean=prior_mean, prior_theta=prior_theta,
                      ess_min=ess_min,
                      ess_max=ess_max,
                      verbose=verbose)

post_clas = post_clas.perform_smc()
with open(f'{folder_name}/sol_clas.pkl', 'wb') as f:
    pickle.dump(post_clas, f)

# %% LOADING SOLUTIONS
with open(f'{folder_name}/sol.pkl', 'rb') as f:
    post = pickle.load(f)
with open(f'{folder_name}/sol_clas.pkl', 'rb') as f:
    post_clas = pickle.load(f)

# %% PLOT PARAMETERS
post.plot_marginals()
post_clas.plot_marginals()
