import copy
import math as mat
import numpy as np
import scipy.stats as stats

def log_normal(x, mean, std):
    return -np.log(mat.sqrt(2 * np.pi) * std) - 0.5 * np.square((x - mean) / std)

def resampling(parameters):
    if parameters.ess[-1] < 0.5 * parameters.n_particles:
        parameters.ess[-1] = parameters.n_particles
        auxiliary_particle = copy.deepcopy(parameters.particle)
        u = np.random.rand()
        for idx, _p in enumerate(parameters.particle):
            threshold = (u + idx) / parameters.n_particles
            sum_weight = 0
            j = -1
            while sum_weight < threshold and j < parameters.n_particles - 1:
                j += 1
                sum_weight += parameters.particle[j].weight
            parameters.particle[idx] = copy.deepcopy(auxiliary_particle[j])
        for _p in parameters.particle:
            _p.weight = 1 / parameters.n_particles
            _p.weight_unnorm = parameters.norm_cost[-1] / parameters.n_particles

    return parameters

def online_estimates(parameters):
    mod_sel = np.zeros(parameters.max_num + 1)
    for i in range(0, len(parameters.particle)):
        mod_sel[parameters.particle[i].num] += parameters.particle[i].weight

    return mod_sel

def sequence_of_exponents(n_iter, max_exp):
    return np.concatenate((np.power(np.linspace(0, 1, n_iter), 4), [max_exp + 0.1]))

def creation_data(n_data, noise_std):
    sourcespace = np.linspace(-5, 5, n_data)
    data = np.zeros(int(n_data))
    for i in range(0, n_data):
        data[i] = 0.7 * stats.norm.pdf(sourcespace[i], -2, 1) + \
                    0.3 * stats.norm.pdf(sourcespace[i], 2, 0.5) + \
                    np.random.normal(0, noise_std)
    return sourcespace, data

def store_iteration(parameters):
    parameters.all_particles = np.concatenate((parameters.all_particles,[parameters.particle]), axis=0)
    parameters.all_weights_unnorm = np.concatenate((parameters.all_weights_unnorm,[np.array([_p.weight_unnorm for _p in parameters.particle])]), axis=0)
    parameters.all_weights = np.concatenate((parameters.all_weights,[np.array([_p.weight for _p in parameters.particle])]), axis=0)
    parameters.mod_sel = np.concatenate((parameters.mod_sel,[online_estimates(parameters)]), axis=0)

    return parameters

def vector_parameters(parameters):
    parameters.vector_mean = []
    parameters.vector_std = []
    parameters.vector_amp = []
    parameters.vector_noise_std = []
    parameters.vector_weight = []
    for _p in parameters.particle:
        for j in range(_p.num):
            parameters.vector_mean.append(_p.gaussian[j].mean)
            parameters.vector_std.append(_p.gaussian[j].std)
            parameters.vector_amp.append(_p.gaussian[j].amp)
            parameters.vector_noise_std.append(_p.noise_std)
            parameters.vector_weight.append(_p.weight / _p.num)

    return parameters

def evaluation_delta(noise_std, exponent_like):
    noise_std_sample = np.divide(noise_std, np.sqrt(exponent_like))
    delta_std = np.zeros(len(noise_std_sample))
    delta_std[0] = abs(noise_std_sample[0] - noise_std_sample[1])
    delta_std[-1] = abs(noise_std_sample[-2] - noise_std_sample[-1])
    for i in range(2, len(noise_std_sample)):
        delta_std[i - 1] = abs(noise_std_sample[i - 2] - noise_std_sample[i])

    return delta_std