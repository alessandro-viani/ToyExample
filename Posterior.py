import copy
import time

import numpy as np
import scipy

from Particle import Particle, mean_prior, theta_prior, evaluation_likelihood
from Util import sequence_of_exponents, bin_creation


class Posterior(object):

    def __init__(self, cfg=None):

        if cfg is None:
            cfg = []

        if 'n_particles' not in cfg:
            self.n_particles = 10
        else:
            self.n_particles = int(cfg['n_particles'])

        if 'theta_eff' not in cfg:
            print('Error: set an estimate for the noise standard deviation')
        else:
            self.theta_eff = cfg['theta_eff']

        if 'sourcespace' not in cfg:
            print('Error: set a sourcespace')
        else:
            self.sourcespace = cfg['sourcespace']

        if 'data' not in cfg:
            print('Error: set data')
        else:
            self.data = cfg['data']

        if 'n_bins' not in cfg:
            self.n_bins = 100
        else:
            self.n_bins = cfg['n_bins']

        if 'sequence_evolution' not in cfg:
            self.sequence_evolution = None
        else:
            self.sequence_evolution = cfg['sequence_evolution']

        if 'method' not in cfg:
            self.method = False
        else:
            self.method = cfg['method']

        if 'verbose' not in cfg:
            self.verbose = False
        else:
            self.verbose = cfg['verbose']

        if self.verbose:
            print(f'number of particles set at: {self.n_particles}')
            print(f'number of bins for MAP estimate set at: {self.n_bins}')
            if self.sequence_evolution is None:
                print('number of iteration adaptively set')
            else:
                print(f'number of iteration set: {self.sequence_evolution}')
            print(f'method: {self.method}')

        self.exponent_like = np.array([0.0, 0.0])
        self.ess = self.n_particles
        self.norm_cost = 1
        self.particle = np.array([Particle(cfg=cfg) for _ in range(0, self.n_particles)])
        for _p in self.particle:
            _p.weight = 1 / self.n_particles

        self.all_particles = np.array([self.particle])
        self.all_weights_unnorm = np.array([np.ones(self.n_particles)])
        self.all_weights = 1 / self.n_particles * np.array([np.ones(self.n_particles)])

        self.n_iter = None

        self.grid_theta = None
        self.theta_posterior = None
        self.map_theta = None
        self.pm_theta = None
        self.ml_theta = None

        self.map_mean = None
        self.pm_mean = None

        self.cpu_time = None

        self.ess_max = 0.99
        self.ess_min = 0.9
        self.delta_max = 1e-1
        self.delta_min = 1e-3

    def metropolis_hastings(self):
        for idx, _p in enumerate(self.particle):
            self.particle[idx] = self.particle[idx].mh_mean(self.sourcespace, self.data, self.exponent_like[-1])
            if self.method == 'FB' or self.method == 'EM':
                self.particle[idx] = self.particle[idx].mh_theta(self.sourcespace, self.data, self.exponent_like[-1])

        return self

    def importance_sampling(self, next_alpha):
        weight_u = np.zeros(self.n_particles)
        for idx, _p in enumerate(self.particle):
            new_like = evaluation_likelihood(_p.mean, _p.theta, self.sourcespace, self.data, next_alpha)
            if _p.like == 0:
                weight_upgrade = 0
            else:
                weight_upgrade = new_like / _p.like
            weight_u[idx] = _p.weight_u * weight_upgrade
            _p.like = new_like
        weight = np.divide(weight_u, np.sum(weight_u))

        for idx, _p in enumerate(self.particle):
            _p.weight_u = weight_u[idx]
            _p.weight = weight[idx]

        self.ess = np.append(self.ess, 1 / np.sum(np.power(weight, 2)))
        self.norm_cost = np.append(self.norm_cost, 1 / self.n_particles * np.sum(weight_u))

        return self

    def resampling(self):
        if self.ess[-1] < 0.5 * self.n_particles:
            self.ess[-1] = self.n_particles
            auxiliary_particle = copy.deepcopy(self.particle)
            u = np.random.rand()
            for idx, _p in enumerate(self.particle):
                threshold = (u + idx) / self.n_particles
                sum_weight = 0
                j = -1
                while sum_weight < threshold and j < self.n_particles - 1:
                    j += 1
                    sum_weight += self.particle[j].weight
                self.particle[idx] = copy.deepcopy(auxiliary_particle[j])
            for _p in self.particle:
                _p.weight = 1 / self.n_particles
                _p.weight_u = self.norm_cost[-1] / self.n_particles

        return self

    def evolution_exponent(self):
        if self.sequence_evolution is None:
            if self.exponent_like[-1] == 1:
                next_exponent = 1.1
            else:
                delta_a = self.delta_min
                delta_b = self.delta_max
                is_last_operation_increment = False
                delta = self.delta_max
                next_exponent = self.exponent_like[-1] + delta
                self_aux = copy.deepcopy(self)
                self_aux.ess[-1] = 0
                iterations = 1
                while not self.ess_min <= self_aux.ess[-1] / self.ess[-1] <= self.ess_max and iterations < 1e2:
                    self_aux = copy.deepcopy(self)
                    self_aux = self_aux.importance_sampling(next_exponent)

                    if self_aux.ess[-1] / self.ess[-1] > self.ess_max:
                        delta_a = delta
                        delta = min((delta_a + delta_b) / 2, self.delta_max)
                        is_last_operation_increment = True
                        if self.delta_max - delta < self.delta_max / 100:
                            next_exponent = self.exponent_like[-1] + delta
                            self_aux = self_aux.importance_sampling(next_exponent)
                            if next_exponent >= 1:
                                next_exponent = 1
                                self_aux.ess[-1] = self.ess[-1] * (self.ess_max + self.ess_min) / 2
                            break
                    else:
                        if self_aux.ess[-1] / self.ess[-1] < self.ess_min:
                            delta_b = delta
                            delta = max((delta_a + delta_b) / 2, self.delta_min)
                            if delta - self.delta_min < self.delta_min / 10 or \
                                    (iterations > 1 and is_last_operation_increment):
                                next_exponent = self.exponent_like[-1] + delta
                                self_aux = self_aux.importance_sampling(next_exponent)
                                if next_exponent >= 1:
                                    next_exponent = 1
                                    self_aux.ess[-1] = self.ess[-1] * (self.ess_max + self.ess_min) / 2
                                break
                                is_last_operation_increment = False
                    next_exponent = self.exponent_like[-1] + delta
                    if next_exponent >= 1:
                        next_exponent = 1
                        self_aux.ess[-1] = self.ess[-1] * (self.ess_max + self.ess_min) / 2
                    iterations += 1
        else:
            next_exponent = sequence_of_exponents(self.sequence_evolution, 1)[len(self.exponent_like)]

        return next_exponent

    def perform_smc(self):
        start_time = time.time()

        if self.method == 'EM':
            self.map_theta_eval()

        n = 0
        if self.verbose:
            print(f'iter:{n} -- exp: {self.exponent_like[n]}')

        self = self.importance_sampling(self.exponent_like[-1])

        self.all_particles = np.concatenate([self.all_particles, np.array([self.particle])], axis=0)
        self.all_weights_unnorm = np.concatenate([self.all_weights_unnorm,
                                                  np.array([[_p.weight_u for _p in self.particle]])],
                                                 axis=0)
        self.all_weights = np.concatenate([self.all_weights,
                                           np.array([[_p.weight for _p in self.particle]])])

        n = 1
        while self.exponent_like[-1] <= 1:
            self = self.metropolis_hastings()
            self.exponent_like = np.append(self.exponent_like, self.evolution_exponent())
            self = self.importance_sampling(self.exponent_like[-1])
            self = self.resampling()
            self.vector_post()
            self.store_iteration()
            self.mean_estimates()
            if self.verbose:
                print(f'iter:{n} -- exp: {"{:.4f}".format(self.exponent_like[n])}')
                if self.method == 'FB' or self.method == 'EM':
                    print(f'MAP mean: {self.map_mean} -- PM mean: {self.pm_mean}')
            n += 1
        self.n_iter = n

        if self.method == 'PM':
            self = self.compute_big_posterior()
            self.vector_post()
            self.mean_estimates()
            if self.verbose:
                print(f'MAP mean: {self.map_mean} -- PM mean: {self.pm_mean}')

        self.theta_estimates()
        if self.verbose:
            print(f'MAP theta: {self.map_theta} -- PM theta: {self.pm_theta}')

        self.cpu_time = time.time() - start_time
        if self.verbose:
            print('\n-- time for execution: %s (s) --' % self.cpu_time)

        return self

    def map_theta_eval(self):
        self.grid_theta = np.linspace(0.5 * self.theta_eff, 10 * self.theta_eff, 100)
        self.theta_posterior = np.zeros(len(self.grid_theta))
        for j, _t in enumerate(self.grid_theta):
            self.theta_posterior[j] = self.integral(theta=_t) * scipy.stats.gamma.pdf(_t, a=2,
                                                                                      scale=4 * self.grid_theta[0])
        delta = np.abs(self.grid_theta[:-1] - self.grid_theta[1:])
        integral = 0.5 * np.sum((self.theta_posterior[:-1] + self.theta_posterior[1:]) * delta)
        self.theta_posterior /= integral
        self.theta_eff = self.grid_theta[np.argmax(self.theta_posterior)]

    def integral(self, theta):
        res = 0
        for _m in self.sourcespace:
            res += evaluation_likelihood(_m, theta, self.sourcespace, self.data, exponent_like=1) * mean_prior(_m)
        return 1 / len(self.sourcespace) * res

    def compute_big_posterior(self):
        particle_aux = []
        integral_weight_u = []
        norm_cost = self.norm_cost[2: self.n_iter]
        exponent_like = self.exponent_like[2: self.n_iter]

        all_theta = self.theta_eff / np.sqrt(exponent_like)

        delta_std = np.zeros(len(all_theta))
        delta_std[0] = abs(all_theta[0] - all_theta[1])
        delta_std[-1] = abs(all_theta[-2] - all_theta[-1])
        for i in range(2, len(all_theta)):
            delta_std[i - 1] = abs(all_theta[i - 2] - all_theta[i])
        k = np.power(np.power(2 * np.pi * np.square(self.theta_eff), exponent_like - 1) * exponent_like, self.data.shape[0] / 2)
        weight_upgrade = 0.5 * delta_std * k * theta_prior(all_theta, self.theta_eff) * norm_cost

        for t_idx in range(self.n_iter - 2):
            for p_idx in range(self.n_particles):
                integral_weight_u = np.append(integral_weight_u, self.all_weights[t_idx + 2, p_idx] * weight_upgrade[t_idx])
                particle_aux = np.append(particle_aux, self.all_particles[t_idx + 2, p_idx])

        integral_weight = integral_weight_u / np.sum(integral_weight_u)
        self.particle = copy.deepcopy(particle_aux)
        for idx, _p in enumerate(self.particle):
            _p.weight_u = integral_weight_u[idx]
            _p.weight = integral_weight[idx]

        self.ess = 1 / np.sum(integral_weight ** 2)
        self.ml_theta = all_theta[np.argmax(norm_cost * k)]
        self.theta_posterior = theta_prior(all_theta, self.theta_eff) * norm_cost * k
        self.grid_theta = np.unique(np.sort(np.append(all_theta, np.linspace(np.min(all_theta), np.max(all_theta), int(1e4)))))
        self.theta_posterior = scipy.interpolate.interp1d(all_theta, self.theta_posterior, kind='linear')(self.grid_theta)
        integral = 0.5 * np.sum((self.theta_posterior[:-1] + self.theta_posterior[1:]) * np.abs(self.grid_theta[:-1] - self.grid_theta[1:]))
        self.theta_posterior /= integral

        return self

    def mean_estimates(self):
        self.pm_mean = 0
        for _p in self.particle:
            self.pm_mean += _p.mean * _p.weight

        left_bin, center_bin, right_bin = bin_creation(np.min(self.sourcespace), np.max(self.sourcespace), self.n_bins)
        weight_bin = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            for idx, _p in enumerate(self.particle):
                if left_bin[i] <= _p.mean <= right_bin[i]:
                    weight_bin[i] += _p.weight
        self.map_mean = center_bin[np.argmax(weight_bin)]

    def theta_estimates(self):
        n_bins = int(0.5*self.n_bins)
        self.pm_theta = 0
        if self.method == 'PM' or self.method == 'EM':
            # posterior mean
            delta = np.abs(self.grid_theta[:-1] - self.grid_theta[1:])
            self.pm_theta = 0.5 * np.sum((self.grid_theta[:-1] * self.theta_posterior[:-1] + self.grid_theta[1:] * self.theta_posterior[1:]) * delta)
        if self.method == 'PM':
            # maximum a posteriori
            self.map_theta = self.grid_theta[np.argmax(self.theta_posterior)]

        if self.method == 'EM':
            # maximum a posteriori
            self.map_theta = self.theta_eff

        if self.method == 'FB':
            # posterior mean
            self.pm_theta = np.sum(np.array(self.vector_weight) * np.array(self.vector_theta))

            # maximum a posteriori
            left_bin_theta, center_bin_theta, right_bin_theta = bin_creation(np.min(self.vector_theta), np.max(self.vector_theta), n_bins)
            weight_bin_theta = np.zeros(n_bins)
            for idx in range(n_bins):
                for jdx, _n in enumerate(self.vector_theta):
                    if left_bin_theta[idx] <= _n <= right_bin_theta[idx]:
                        weight_bin_theta[idx] += self.vector_weight[jdx]
            self.map_theta = center_bin_theta[np.argmax(weight_bin_theta)]

    def vector_post(self):
        self.vector_mean = []
        self.vector_theta = []
        self.vector_weight = []
        self.vector_weight_u = []
        for _p in self.particle:
            self.vector_mean.append(_p.mean)
            self.vector_theta.append(_p.theta)
            self.vector_weight.append(_p.weight)
            self.vector_weight_u.append(_p.weight_u)

    def store_iteration(self):
        self.all_particles = np.concatenate([self.all_particles, np.array([self.particle])], axis=0)
        self.all_weights_unnorm = np.concatenate([self.all_weights_unnorm, np.array([self.vector_weight_u])], axis=0)
        self.all_weights = np.concatenate([self.all_weights, np.array([self.vector_weight])], axis=0)
