import pickle

import numpy as np
import seaborn as sns

from Posterior import Posterior
from Util import creation_data

n_times = 100
n_data = 20
n_particles = 10
folder_name = 'sol/'

# prior parameters
prior_mean = [-5, 5]
prior_theta = [2, 4]
theta_true = 0.1
ml_theta = np.zeros(n_times)

# %%
for j in range(n_times):
    sourcespace, data = creation_data(n_data=n_data, theta=theta_true)
    with open(f'data/data_unbiased_{j}.pkl', 'wb') as f:
        pickle.dump([sourcespace, data, theta_true], f)

for j in range(n_times):
    print(f'iter:{j}')
    with open(f'data/data_unbiased_{j}.pkl', 'rb') as f:
        sourcespace, data, useless = pickle.load(f)
    post = Posterior(prop_method=True,
                     sourcespace=sourcespace,
                     data=data,
                     n_particles=n_particles,
                     theta_eff=theta_true / 2,
                     prior_mean=prior_mean,
                     prior_theta=prior_theta,
                     max_exp=1,
                     ess_min=0.7,
                     ess_max=0.79,
                     delta_min=1e-3,
                     delta_max=1e-1,
                     verbose=False)

    post = post.perform_smc()

    with open(f'sol/sol_unbiased_{j}.pkl', 'wb') as f:
        pickle.dump(post, f)

# %% ERRORS EVALUATION
for j in range(n_times):
    with open(f'sol/sol_unbiased_{j}.pkl', 'rb') as f:
        post = pickle.load(f)
    ml_theta[j] = post.ml_theta

with open('sol/ml_theta.pkl', 'wb') as f:
    pickle.dump(ml_theta, f)

# %% PLOT
with open('sol/ml_theta.pkl', 'rb') as f:
    ml_theta = pickle.load(f)

sns.histplot(ml_theta - theta_true, bins=10)
print(1 / n_times * np.sum(ml_theta - theta_true))
