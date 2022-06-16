# -*- coding: utf-8 -*-
"""
@author: Alessandro Viani (2022)
"""
import pickle
from Posterior import Posterior
from Util import creation_data

true_noise = 0.1
n_data = 10
folder_name = 'sol/'

#CREATE DATA
sourcespace, data = creation_data(n_data=n_data, noise_std=true_noise)
with open('data/data.pkl', 'wb') as f:
    pickle.dump([sourcespace, data, true_noise], f)

#RUN PROPOSED METHOD
post = Posterior(n_particles=20, theta_eff=true_noise/2, prop_method=True, sequence_evolution=None,
             sourcespace=sourcespace, data=data, prior_mean=[-5, 5], prior_theta=[2, 4],
             max_exp=1, ess_min=0.70, ess_max=0.79, delta_min=1e-3, delta_max=1e-1,
             point_spline=1e4, n_bins=10, verbose=True)

post = post.perform_smc()
post.plot_marginals()
with open(f'{folder_name}/sol_prop.pkl', 'wb') as f:
    pickle.dump(post, f)

#RUN FULLY BAYESIAN
post = Posterior(n_particles=20, theta_eff=true_noise/2, prop_method=False, sequence_evolution=None,
             sourcespace=sourcespace, data=data, prior_mean=[-5, 5], prior_theta=[2, 4],
             max_exp=1, ess_min=0.70, ess_max=0.79, delta_min=1e-3, delta_max=1e-1,
             point_spline=1e4, n_bins=10, verbose=True)

post = post.perform_smc()
post.plot_marginals()
with open(f'{folder_name}/sol_fully_bayesian.pkl', 'wb') as f:
    pickle.dump(post, f)
