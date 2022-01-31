# -*- coding: utf-8 -*-

# Author: Alessandro Viani <viani@dima.unige.it>
#
# License: BSD (3-clause)

import pickle

import numpy as np
import scipy.stats as stats

from SMC import Posterior
from Util import creation_data

n_file = 1
n_data = 20
n_particles = 10
kl_div = np.zeros(n_file)
folder_name = 'performance_analysis/'
noise_std = np.sort(0.1 - (0.1 - 0.02) * np.random.rand(n_file))

for idx, _n in enumerate(noise_std):
    sourcespace, data = creation_data(n_data=n_data, noise_std=_n)
    with open(f'{folder_name}data_{idx}.pkl', 'wb') as f:
        pickle.dump([sourcespace, data, _n], f)

    post_prop = Posterior(num_evolution=None, mean_evolution=True, std_evolution=True,
                          amp_evolution=True, noise_evolution=False, sequence_evolution=None,
                          mh_evolution=False, sourcespace=sourcespace, data=data,
                          max_exp=1, n_particles=n_particles, max_num=10, noise_std_eff=_n / 2,
                          prior_num=0.25, prior_m=[-5, 5], prior_s=[0.1, 10], prior_a=[1, 0.25], prior_n=[2, 4],
                          prop_method=True)

    post_prop = post_prop.perform_smc()

    with open(f'{folder_name}posterior_proposed_{idx}.pkl', 'wb') as f:
        pickle.dump(post_prop, f)

    post_clas = Posterior(num_evolution=None, mean_evolution=True, std_evolution=True,
                          amp_evolution=True, noise_evolution=True, sequence_evolution=None,
                          mh_evolution=False, sourcespace=sourcespace, data=data,
                          max_exp=1, n_particles=n_particles, max_num=10, noise_std_eff=_n / 2,
                          prior_num=0.25, prior_m=[-5, 5], prior_s=[0.1, 10], prior_a=[1, 0.25], prior_n=[2, 4],
                          prop_method=False)

    post_clas = post_clas.perform_smc()
    post_clas.noise_posterior = stats.gaussian_kde(post_clas.vector_noise_std,
                                                   weights=post_clas.vector_weight).pdf(post_prop.all_noise_std)

    with open(f'{folder_name}posterior_classical_{idx}.pkl', 'wb') as f:
        pickle.dump(post_clas, f)
