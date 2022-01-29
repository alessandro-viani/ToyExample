import copy
import time

import numpy as np

from Particle import Particle
from Util import sequence_of_exponents


class Posterior(object):
    def __init__(self, num_evolution=None, mean_evolution=True, std_evolution=True,
                 amp_evolution=True, noise_evolution=False, sequence_evolution=True,
                 mh_evolution=False, sourcespace=None, data=None,
                 max_exp=1, n_particles=100, max_num=3, noise_std_eff=None,
                 lam=0.25, prior_m=[-5, 5], prior_s=[0.1, 10], prior_a=[1, 0.25], prior_n=[2, 4], prop_method=True):

        self.prop_method = prop_method
        self.num_evolution = num_evolution
        self.mean_evolution = mean_evolution
        self.std_evolution = std_evolution
        self.amp_evolution = amp_evolution

        self.noise_evolution = noise_evolution

        self.sequence_evolution = sequence_evolution
        self.mh_evolution = mh_evolution

        self.lam = lam
        self.prior_m = prior_m
        self.prior_s = prior_s
        self.prior_a = prior_a
        self.prior_n = prior_n

        self.sourcespace = sourcespace
        self.data = data

        self.max_exp = max_exp
        self.n_particles = n_particles
        self.max_num = max_num
        if self.sequence_evolution is None:
            self.exponent_like = np.array([0, 0])
        else:
            self.exponent_like = sequence_of_exponents(self.sequence_evolution, self.max_exp)

        self.noise_std_eff = noise_std_eff

        self.ess = self.n_particles
        self.norm_cost = self.n_particles

        self.mod_sel = []
        self.all_particles = []
        self.all_weights_unnorm = []
        self.all_weights = []

        self.particle = np.array([Particle(num=self.inizialize_num(), noise_std_eff=self.noise_std_eff,
                                           num_evolution=self.num_evolution,
                                           mean_evolution=self.mean_evolution, std_evolution=self.std_evolution,
                                           amp_evolution=self.amp_evolution, noise_evolution=self.noise_evolution,
                                           lam=self.lam,
                                           prior_m=self.prior_m,
                                           prior_s=self.prior_s,
                                           prior_a=self.prior_a,
                                           prior_n=self.prior_n)
                                  for i in range(0, self.n_particles)])
        for _p in self.particle:
            _p.weight = 1 / self.n_particles

    def inizialize_num(self):
        if self.num_evolution is None:
            num = np.random.poisson(0.25)
            while num > self.max_num:
                num = np.random.poisson(0.25)
        else:
            num = self.num_evolution
        return num

    def metropolis_hastings(self):
        for idx, _p in enumerate(self.particle):
            if self.num_evolution is None:
                self.particle[idx] = self.particle[idx].mh_num(self)
            if self.mean_evolution:
                self.particle[idx] = self.particle[idx].mh_mean(self)
            if self.std_evolution:
                self.particle[idx] = self.particle[idx].mh_std(self)
            if self.amp_evolution:
                self.particle[idx] = self.particle[idx].mh_amp(self)
            if self.noise_evolution:
                self.particle[idx] = self.particle[idx].mh_noise(self)
        return self

    def importance_sampling(self, next_alpha):
        weight_unnorm = np.zeros(self.n_particles)
        for idx, _p in enumerate(self.particle):
            new_like = _p.evaluation_likelihood(self.sourcespace, self.data, next_alpha)
            if _p.like == 0:
                print('warning: particle with zero likelihood')
                weight_upgrade = 0
            else:
                weight_upgrade = new_like / _p.like
            weight_unnorm[idx] = _p.weight_unnorm * weight_upgrade
            _p.like = new_like
        weight = np.divide(weight_unnorm, np.sum(weight_unnorm))

        for idx, _p in enumerate(self.particle):
            _p.weight_unnorm = weight_unnorm[idx]
            _p.weight = weight[idx]

        self.ess = np.append(self.ess, 1 / np.sum(np.power(weight, 2)))
        self.norm_cost = np.append(self.norm_cost, np.sum(weight_unnorm))

        return self

    def evolution_exponent(self):

        if self.sequence_evolution is None:
            if self.exponent_like[-1] == self.max_exp:
                next_exponent = self.max_exp + 0.1
            else:
                ess_max = 0.99
                ess_min = 0.9
                delta_max = 0.1
                delta_min = 1e-3
                delta_a = delta_min
                delta_b = 0.1
                is_last_operation_increment = False
                delta = delta_max
                next_exponent = self.exponent_like[-1] + delta
                self_aux = copy.deepcopy(self)
                self_aux.ess[-1] = 0
                iterations = 1
                while not ess_min <= self_aux.ess[-1] / self.ess[-1] <= ess_max and iterations < 1e2:
                    self_aux = copy.deepcopy(self)
                    self_aux = self_aux.importance_sampling(next_exponent)

                    if self_aux.ess[-1] / self.ess[-1] > ess_max:
                        delta_a = delta
                        delta = min((delta_a + delta_b) / 2, delta_max)
                        is_last_operation_increment = True
                        if delta_max - delta < delta_max / 100:
                            next_exponent = self.exponent_like[-1] + delta
                            self_aux = self_aux.importance_sampling(next_exponent)
                            if next_exponent >= 1:
                                next_exponent = 1
                                self_aux.ess[-1] = self.ess[-1] * (ess_max + ess_min) / 2
                            break
                    else:
                        if self_aux.ess[-1] / self.ess[-1] < ess_min:
                            delta_b = delta
                            delta = max((delta_a + delta_b) / 2, delta_min)
                            if delta - delta_min < delta_min / 10 or \
                                    (iterations > 1 and is_last_operation_increment == True):
                                next_exponent = self.exponent_like[-1] + delta
                                self_aux = self_aux.importance_sampling(next_exponent)
                                if next_exponent >= 1:
                                    next_exponent = 1
                                    self_aux.ess[-1] = self.ess[-1] * (ess_max + ess_min) / 2
                                break
                                is_last_operation_increment = False
                    next_exponent = self.exponent_like[-1] + delta
                    if next_exponent >= 1:
                        next_exponent = 1
                        self_aux.ess[-1] = self.ess[-1] * (ess_max + ess_min) / 2
                    iterations += 1
        else:
            next_exponent = sequence_of_exponents(self.sequence_evolution,
                                                  self.max_exp)[len(self.exponent_like)]

        return next_exponent

    def online_estimates(self):
        mod_sel = np.zeros(self.max_num + 1)
        for i in range(0, len(self.particle)):
            mod_sel[self.particle[i].num] += self.particle[i].weight

        return mod_sel

    def perform_smc(self):
        start_time = time.time()
        n = 0
        print(f'iter:{n} -- exp: {self.exponent_like[n]} -- est num: 0')

        self.all_particles = copy.deepcopy(self.particle)
        self.all_weights_unnorm = np.array([_p.weight_unnorm for _p in self.particle])
        self.all_weights = np.array([_p.weight for _p in self.particle])
        self.mod_sel = self.online_estimates()

        self = self.importance_sampling(self.exponent_like[-1])

        self.all_particles = np.concatenate([self.all_particles, self.particle])
        self.all_particles = self.all_particles.reshape(2, self.n_particles)

        self.all_weights_unnorm = np.concatenate(
            [self.all_weights_unnorm, np.array([_p.weight_unnorm for _p in self.particle])])
        self.all_weights_unnorm = self.all_weights_unnorm.reshape(2, self.n_particles)

        self.all_weights = np.concatenate([self.all_weights, np.array([_p.weight for _p in self.particle])])
        self.all_weights = self.all_weights.reshape(2, self.n_particles)

        self.mod_sel = np.concatenate([self.mod_sel, self.online_estimates()])
        self.mod_sel = self.mod_sel.reshape(2, self.max_num + 1)

        n = 1
        while self.exponent_like[-1] <= self.max_exp:
            diff = 1
            while diff > 1e-2:
                self = self.metropolis_hastings()
                if self.sequence_evolution is None:
                    self.exponent_like = np.append(self.exponent_like, self.evolution_exponent())
                if n > 1 and self.mh_evolution:
                    particle_aux_is = copy.deepcopy(self.particle)
                    old_norm_cost = self.norm_cost[n + 1]
                    self = 1  # importance_sampling(particle_aux_is, self)
                    diff = abs(self.norm_cost[n + 1] - old_norm_cost) / old_norm_cost
                else:
                    diff = 0
                    self = self.importance_sampling(self.exponent_like[-1])
            self = self.resampling()
            self = self.store_iteration()
            print(f'iter:{n} -- exp: {self.exponent_like[n]} -- est num: {np.argmax(self.mod_sel[n][:])}')
            if self.mh_evolution:
                self.particle = copy.deepcopy(particle_aux_is)
            n += 1
        self.n_iter = n
        self.vector_post()
        self.est_num = np.argmax(self.mod_sel[n][:])
        print("\n-- time for execution: %s (s) --" % (time.time() - start_time))

        if self.prop_method:
            self = self.compute_big_posterior()

        return self

    def compute_big_posterior(self):
        n_iter = self.n_iter
        norm_cost = self.norm_cost[2: n_iter]
        exponent_like = self.exponent_like[2: n_iter]
        self.all_noise_std = self.noise_std_eff / np.sqrt(exponent_like)

        noise_std_sample = np.divide(self.all_noise_std, np.sqrt(exponent_like))
        delta_std = np.zeros(len(noise_std_sample))
        delta_std[0] = abs(noise_std_sample[0] - noise_std_sample[1])
        delta_std[-1] = abs(noise_std_sample[-2] - noise_std_sample[-1])
        for i in range(2, len(noise_std_sample)):
            delta_std[i - 1] = abs(noise_std_sample[i - 2] - noise_std_sample[i])

        weight = self.all_weights[2:, :]

        self.particle_avg = []
        integral_weight_u = []

        weight_upgrade = np.zeros(n_iter - 2)
        k = np.zeros(n_iter - 2)
        prior = self.particle[0].noise_prior(self.all_noise_std)
        k = np.power(np.power(2 * np.pi * np.square(self.noise_std_eff), exponent_like - 1) * exponent_like,
                     self.data.shape[0] / 2)
        weight_upgrade = 0.5 * delta_std * k * prior * norm_cost
        for i in range(n_iter - 2):
            for j in range(self.n_particles):
                integral_weight_u = np.append(integral_weight_u, weight[i, j] * weight_upgrade[i])
                self.particle_avg = np.append(self.particle_avg, self.all_particles[i, j])

        integral_weight = integral_weight_u / np.sum(integral_weight_u)

        for i, _p in enumerate(self.particle_avg):
            _p.weight_unnorm = integral_weight_u[i]
            _p.weight = integral_weight[i]

        self.noise_posterior = prior * norm_cost * k / np.sum(prior * norm_cost * k)

        mod_sel_avg = np.zeros(self.max_num + 1)
        for i in range(0, len(self.particle_avg)):
            mod_sel_avg[self.particle_avg[i].num] += self.particle_avg[i].weight

        self.est_num_avg = np.argmax(mod_sel_avg)

        return self

    def vector_post(self):
        self.vector_mean = []
        self.vector_std = []
        self.vector_amp = []
        self.vector_noise_std = []
        self.vector_weight = []
        for _p in self.particle:
            for j in range(_p.num):
                self.vector_mean.append(_p.gaussian[j].mean)
                self.vector_std.append(_p.gaussian[j].std)
                self.vector_amp.append(_p.gaussian[j].amp)
                self.vector_noise_std.append(_p.noise_std)
                self.vector_weight.append(_p.weight / _p.num)

    def store_iteration(self):
        self.all_particles = np.concatenate((self.all_particles, [self.particle]), axis=0)
        self.all_weights_unnorm = np.concatenate(
            (self.all_weights_unnorm, [np.array([_p.weight_unnorm for _p in self.particle])]), axis=0)
        self.all_weights = np.concatenate((self.all_weights, [np.array([_p.weight for _p in self.particle])]), axis=0)
        self.mod_sel = np.concatenate((self.mod_sel, [self.online_estimates()]), axis=0)

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
                _p.weight_unnorm = self.norm_cost[-1] / self.n_particles

        return self
