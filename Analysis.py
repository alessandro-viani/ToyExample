# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:55:32 2022

@author: viani
"""

import pickle
from Posterior import Posterior
from Util import creation_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_file = 100
n_data = 100
n_particles = 100
verbose = True
ess_min = 0.7
ess_max = 0.79
folder_name = 'sol/'
theta_true = np.linspace(0.01, 0.1, n_file)

err_map_mean = np.zeros(n_file)
err_map_theta = np.zeros(n_file)
err_pm_mean = np.zeros(n_file)
err_pm_theta = np.zeros(n_file)
err_ml_theta = np.zeros(n_file)

err_map_mean_clas = np.zeros(n_file)
err_map_theta_clas = np.zeros(n_file)
err_pm_mean_clas = np.zeros(n_file)
err_pm_theta_clas = np.zeros(n_file)

cpu_time = np.zeros(n_file)
cpu_time_clas = np.zeros(n_file)
ess = np.zeros(n_file)
ess_clas = np.zeros(n_file)

for idx, _n in enumerate(theta_true):
    sourcespace, data = creation_data(n_data=n_data, theta=_n)
    with open(f'data/data_prop_vs_full_{idx}.pkl', 'wb') as f:
        pickle.dump([sourcespace, data, _n], f)

# %% RUN SMC SAMPLERS
for idx, _n in enumerate(theta_true):
    print(f'iter:{idx}')
    with open(f'data/data_prop_vs_full_{idx}.pkl', 'rb') as f:
        sourcespace, data, useless = pickle.load(f)
    post = Posterior(prop_method=True, sourcespace=sourcespace, data=data,
                     n_particles=n_particles, theta_eff=_n / 2,
                     ess_min=ess_min, ess_max=ess_max, verbose=verbose)

    post = post.perform_smc()

    with open(f'{folder_name}/sol_prop_vs_full_{idx}.pkl', 'wb') as f:
        pickle.dump(post, f)

    post_clas = Posterior(prop_method=False,
                          sourcespace=sourcespace, data=data,
                          n_particles=n_particles, theta_eff=_n / 2,
                          ess_min=ess_min, ess_max=ess_max, verbose=verbose)

    post_clas = post_clas.perform_smc()
    with open(f'{folder_name}/sol_clas_prop_vs_full_{idx}.pkl', 'wb') as f:
        pickle.dump(post_clas, f)

# %% ERRORS EVALUATION
for idx, _n in enumerate(theta_true):
    print(f'iter:{idx}')

    with open(f'{folder_name}/sol_prop_vs_full_{idx}.pkl', 'rb') as f:
        post = pickle.load(f)
    with open(f'{folder_name}/sol_clas_prop_vs_full_{idx}.pkl', 'rb') as f:
        post_clas = pickle.load(f)

    err_map_mean[idx] = (post.map_mean - 0)
    err_map_mean_clas[idx] = (post_clas.map_mean - 0)

    err_pm_mean[idx] = (post.pm_mean - 0)
    err_pm_mean_clas[idx] = (post_clas.pm_mean - 0)

    err_map_theta[idx] = (post.map_theta - _n) / _n
    err_map_theta_clas[idx] = (post_clas.map_theta - _n) / _n

    err_pm_theta[idx] = (post.pm_theta - _n) / _n
    err_pm_theta_clas[idx] = (post_clas.pm_theta - _n) / _n

    err_ml_theta[idx] = np.abs(post.ml_theta - _n) / _n

    cpu_time[idx] = post.cpu_time
    cpu_time_clas[idx] = post_clas.cpu_time

    ess[idx] = post.ess_big[-1]
    ess_clas[idx] = post_clas.ess[-1]

with open(f'{folder_name}/analytics.pkl', 'wb') as f:
    pickle.dump([err_map_mean, err_map_theta,
                 err_pm_mean, err_pm_theta, err_ml_theta,
                 err_map_mean_clas, err_map_theta_clas,
                 err_pm_mean_clas, err_pm_theta_clas,
                 cpu_time, cpu_time_clas,
                 ess, ess_clas], f)

# %%
saturation = 0.7
linewidth = 1
width = 0.7
dpi = 50
plot_show = True
size_ticks = 22

with open(f'{folder_name}/analytics.pkl', 'rb') as f:
    err_map_mean, err_map_theta, \
    err_pm_mean, err_pm_theta, err_ml_theta, \
    err_map_mean_clas, err_map_theta_clas, \
    err_pm_mean_clas, err_pm_theta_clas, \
    cpu_time, cpu_time_clas, \
    ess, ess_clas = pickle.load(f)

fig, ax = plt.subplots(2, 1, figsize=(16, 9), dpi=100)
data = [err_map_mean_clas, err_map_mean, err_pm_mean_clas, err_pm_mean]
sns.boxplot(data=data,
            palette=['red', '#1f77b4', 'red', '#1f77b4'],
            saturation=saturation,
            width=width,
            linewidth=linewidth,
            ax=ax[0])
plt.sca(ax[0])
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel(None)
plt.xlabel(None)
plt.xticks([0, 1, 2, 3], ['map_clas', 'map_prop', 'pm_clas', 'pm_prop'])
plt.title(r'Error $\mu$')
# plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

sns.boxplot(data=[err_map_theta_clas, err_map_theta, err_pm_theta_clas, err_pm_theta],
            palette=['red', '#1f77b4', 'red', '#1f77b4'],
            saturation=saturation,
            width=width,
            linewidth=linewidth,
            ax=ax[1])
plt.sca(ax[1])
plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel(None)
plt.xlabel(None)
plt.xticks([0, 1, 2, 3], ['map_clas', 'map_prop', 'pm_clas', 'pm_prop'])
plt.title(r'Error $\theta$')
# plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tight_layout()
plt.savefig('fig/sol_proposed_vs_fully_bayesian.png', dpi=100)
# plt.close()

plt.figure(figsize=(16, 9), dpi=100)
sns.boxplot(data=[cpu_time_clas, cpu_time], palette=['red', '#1f77b4'],
            saturation=saturation, width=width,
            linewidth=linewidth)
plt.title('CPU time')
plt.tight_layout()
plt.savefig('fig/cpu_time.png', dpi=dpi)

plt.figure(figsize=(16, 9), dpi=100)
sns.boxplot(data=[ess_clas, ess], palette=['red', '#1f77b4'],
            saturation=saturation, width=width,
            linewidth=linewidth)
plt.title('ESS')
plt.tight_layout()
plt.savefig('fig/ess.png', dpi=dpi)
