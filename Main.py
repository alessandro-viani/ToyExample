# -*- coding: utf-8 -*-

# Author: Alessandro Viani <viani@dima.unige.it>
#
# License: BSD (3-clause)

import pickle

from SMC import Posterior
from Util import creation_data

true_noise_std = 0.05
sourcespace, data = creation_data(n_data=100, noise_std=true_noise_std)
with open('save_folder/data.pkl', 'wb') as f:
    pickle.dump([sourcespace, data], f)

post = Posterior(num_evolution=None, mean_evolution=True, std_evolution=True,
                 amp_evolution=True, prop_method=True, sequence_evolution=None,
                 mh_evolution=False, sourcespace=sourcespace, data=data,
                 max_exp=1, n_particles=1000, max_num=10, noise_std_eff=true_noise_std / 2,
                 prior_num=0.25, prior_m=[-5, 5], prior_s=[0.1, 10], prior_a=[1, 0.25], prior_n=[2, 4])

post = post.perform_smc()

with open('save_folder/posterior.pkl', 'wb') as f:
    pickle.dump(post, f)

post.plot_data()
post.plot_marginals()
