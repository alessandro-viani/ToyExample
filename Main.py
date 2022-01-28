import copy
import pickle
import pickle
import matplotlib.pyplot as plt
from Util import vector_parameters
from ClassSMC import ClassParameters
from SMC import perform_smc, compute_big_posterior
from Util import creation_data
import random

true_noise_std = 0.05
sourcespace, data = creation_data(n_data=10, noise_std=true_noise_std)

parameters = ClassParameters(num_evolution=None, mean_evolution=True, std_evolution=True,
                 amp_evolution=True, noise_evolution=False, sequence_evolution=None,
                 mh_evolution=False, sourcespace=sourcespace, data=data,
                 max_exp=1, n_particles=10, max_num=3, noise_std_eff=true_noise_std/2)

parameters = perform_smc(parameters)
parameters = compute_big_posterior(parameters)

with open('results/solution.pkl', 'wb') as f:
    pickle.dump(parameters, f)

with open(f'results/solution.pkl', 'rb') as f:
    parameters = pickle.load(f)
    
parameters = vector_parameters(parameters)
plt.hist(parameters.vector_mean, weights=parameters.vector_weight, bins=50)
plt.show()
plt.plot(parameters.all_noise_std, parameters.noise_posterior)
plt.show()