# -*- coding: utf-8 -*-

# Author: Alessandro Viani <viani@dima.unige.it>
#
# License: BSD (3-clause)

import pickle

import numpy as np
import scipy.stats as stats

from SMC import Posterior
from Util import creation_data, analytics

n_file = 100
noise_range = [0.02, 0.1]
n_data = 100
n_particles = 1000

kl_div = np.zeros(n_file)
folder_name = 'save_folder/'
noise_std = np.sort(noise_range[1] - (noise_range[1] - noise_range[0]) * np.random.rand(n_file))
for idx, _n in enumerate(noise_std):
    sourcespace, data = creation_data(n_data=n_data, noise_std=_n)
    with open(f'{folder_name}data_{idx}.pkl', 'wb') as f:
        pickle.dump([sourcespace, data, _n], f)

    post_prop = Posterior(num_evolution=None, mean_evolution=True, std_evolution=True,
                          amp_evolution=True, prop_method=True, sequence_evolution=None,
                          mh_evolution=False, sourcespace=sourcespace, data=data,
                          max_exp=1, n_particles=n_particles, max_num=10, noise_std_eff=_n / 2,
                          prior_num=0.25, prior_m=[-5, 5], prior_s=[0.1, 10], prior_a=[1, 0.25], prior_n=[2, 4])

    post_prop = post_prop.perform_smc()

    with open(f'{folder_name}posterior_proposed_{idx}.pkl', 'wb') as f:
        pickle.dump(post_prop, f)

    post_clas = Posterior(num_evolution=None, mean_evolution=True, std_evolution=True,
                          amp_evolution=True, prop_method=False, sequence_evolution=None,
                          mh_evolution=False, sourcespace=sourcespace, data=data,
                          max_exp=1, n_particles=n_particles, max_num=10, noise_std_eff=_n / 2,
                          prior_num=0.25, prior_m=[-5, 5], prior_s=[0.1, 10], prior_a=[1, 0.25], prior_n=[2, 4])

    post_clas = post_clas.perform_smc()
    post_clas.noise_posterior = stats.gaussian_kde(post_clas.vector_noise_std,
                                                   weights=post_clas.vector_weight).pdf(post_prop.all_noise_std)
    post_clas.noise_posterior /= np.sum(post_clas.noise_posterior)

    with open(f'{folder_name}posterior_classical_{idx}.pkl', 'wb') as f:
        pickle.dump(post_clas, f)

analytics(n_file=n_file, interval_mean=[-5, 5], interval_std=[0.1, 10], interval_amp=[0, 10], n_bins=300,
          folder_name=folder_name)