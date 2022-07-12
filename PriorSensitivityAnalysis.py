"""
Created on Jun 2022

@author: ale
"""
import pickle
from Posterior import Posterior
from Util import creation_data
import matplotlib.pyplot as plt
import numpy as np

n_times = 10
theta_true = 0.1
n_data = 5
n_particles = 10
folder_name = 'sol/'

saturation = 0.7
linewidth = 1
width = 0.7
dpi = 50
plot_show = True
size_ticks = 22
n_bins = 50
prior_mean = [-5, 5]

sourcespace, data = creation_data(n_data=n_data, theta=theta_true)
with open('data/data_prior_analysis.pkl', 'wb') as f:
    pickle.dump([sourcespace, data, theta_true], f)
v_prior = np.linspace(1.1, 10, n_times)

post = Posterior(prop_method=True, sourcespace=sourcespace, data=data,
                 n_particles=n_particles, theta_eff=theta_true / 2,
                 prior_mean=prior_mean, prior_theta=[1, 2], verbose=True)
post = post.perform_smc()
with open('sol/sol_prior_analysis.pkl', 'wb') as f:
    pickle.dump(post, f)

# %%
for idx, _n in enumerate(v_prior):
    print(f'iter:{idx}')
    with open('sol/sol_prior_analysis.pkl', 'rb') as f:
        post = pickle.load(f)
    post.prior_theta = [_n, _n + 2]
    post.n_bins = 2
    post.particle_avg = []
    post = post.compute_big_posterior()
    post.theta_estimates()
    with open(f'sol/sol_prior_analysis_{idx}.pkl', 'wb') as f:
        pickle.dump(post, f)

# %% LOADING SOLUTIONS
pm_smc = np.zeros(len(v_prior))
map_smc = np.zeros(len(v_prior))
fig, ax = plt.subplots(3, 1)
for idx, _n in enumerate(v_prior):
    with open(f'sol/sol_prior_analysis_{idx}.pkl', 'rb') as f:
        post = pickle.load(f)

    pm_smc[idx] = post.pm_theta
    map_smc[idx] = post.map_theta
    ax[0].plot(post.grid_theta, post.theta_prior, '#1f77b4')
    ax[1].plot(post.grid_theta, post.theta_posterior, '#1f77b4')

ax[2].plot(v_prior, pm_smc)
ax[2].plot(v_prior, map_smc)
ax[2].legend(['Posteiror Mean', 'MAP'])
fig.tight_layout()
plt.savefig('fig/prior_sensitivity_analysis.png', dpi=100)
plt.show()
