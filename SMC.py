# -*- coding: utf-8 -*-

# Author: Alessandro Viani <viani@dima.unige.it>
#
# License: BSD (3-clause)

import copy
import time

import numpy as np

from Particle import Particle
from Util import sequence_of_exponents


class Posterior(object):
    """Single Particle class for SMC samplers.

       Parameters
       ----------
       n_particles : :py:class:`int`
           The number of particles for performing SMC samplers.
       noise_std_eff : :py:class:`double`
           The estimated noise standard deviation.
       num_evolution : :py:class:`None or int`
           None: the number is one of the parameters to be estimated,
           int: it represents the fixed number of gaussian for each particle.
       mean_evolution : :py:class:`bool`
           True: the mean is one of the parameters to be estimated,
           False: the mean is fixed as the true mean value [works with one gaussian only].
       std_evolution : :py:class:`bool`
           True: the standard deviation is one of the parameters to be estimated,
           False: the standard deviation is fixed as the true standard deviation value [works with one gaussian only].
       amp_evolution : :py:class:`bool`
           True: the amplitude is one of the parameters to be estimated,
           False: the amplitude is fixed as the true amplitude value [works with one gaussian only].
       noise_evolution : :py:class:`bool`
           True: the noise standard deviation is one of the parameters to be estimated,
           False: the noise standard deviation is fixed as the estimated noise standard deviation value.

       sequence_evolution : :py:class:`None or int`
           None: exponent of the sequence are adaptively chosen,
           int: number of iterations of Importance sampling.
       mh_evolution : :py:class:`bool`
           True: Metropolis-Hastings steps are adaptively chosen,
           False: one Metropolis-Hastings step is performed at each iteration.
       prop_method : :py:class:`bool`
           True: at the last iteration particle recycling is performed,
           False: no recycle scheme is performed.
       sourcespace : :py:class:`np.array([double])`
           Grid of x-axes points where data are collected.
       data : :py:class:`np.array([double])`
           Noisy samples form the sum of weighted gaussian functions.
       max_exp : :py:class:`int`
           Maximum exponent reached during the SMC samplers iterations.
       max_num : :py:class:`int`
           Maximum number of gaussian allowed in each particle.

       prior_num : :py:class:`double`
           The mean for the poisson prior on the gaussian number.
       prior_m : :py:class:`np.array([double, double])`
           The interval prior parameters for the uniform prior on gaussian mean.
       prior_s : :py:class:`np.array([double, double])`
           The interval prior parameters for the log-uniform prior on gaussian standard deviation.
       prior_a : :py:class:`np.array([double, double])`
           The mean and standard deviation prior parameters for the normal prior on gaussian amplitude.
       prior_n : :py:class:`np.array([double, double])`
           The shape and scale parameters prior parameters for the gamma prior on noise standard deviation.

        Attributes
        ----------
       ess : :py:class:`np.array([double])`
           The noise standard deviation sampled if noise_evolution is True,
           or the noise standard deviation estimated if noise_evolution is False.
       q_death : :py:class:`double`
           The probability of gaussian death.
       q_birth : :py:class:`double`
           The probability of gaussian birth.
       gaussian : :py:class:`np.array([Gaussian])`
           The array containing particle gaussians.
       like : :py:class:`double`
           The likelihood of the particle.
       prior : :py:class:`double`
           The prior of the particle.
       weight : :py:class:`double`
           The weight of the particle.
       weight_unnorm : :py:class:`double`
           The weight un-normalized of the particle
           [useless if the particle is alone, but useful for performing SMC samplers].
       """
    def __init__(self, n_particles=None, noise_std_eff=None,
                 num_evolution=None, mean_evolution=True, std_evolution=True, amp_evolution=True, noise_evolution=False,
                 sequence_evolution=None, mh_evolution=False , prop_method=True,
                 sourcespace=None, data=None, max_exp=None, max_num=None,
                 prior_num=None, prior_m=None, prior_s=None, prior_a=None, prior_n=None):

        self.n_particles = n_particles
        self.noise_std_eff = noise_std_eff
        self.num_evolution = num_evolution
        self.mean_evolution = mean_evolution
        self.std_evolution = std_evolution
        self.amp_evolution = amp_evolution
        self.noise_evolution = noise_evolution
        self.sequence_evolution = sequence_evolution
        self.mh_evolution = mh_evolution
        self.prop_method = prop_method
        self.sourcespace = sourcespace
        self.data = data
        self.max_exp = max_exp
        self.max_num = max_num
        self.prior_num = prior_num
        self.prior_m = prior_m
        self.prior_s = prior_s
        self.prior_a = prior_a
        self.prior_n = prior_n

        self.exponent_like = np.array([0, 0])
        self.ess = self.n_particles
        self.norm_cost = self.n_particles
        self.mod_sel = []
        self.all_particles = []
        self.all_weights_unnorm = []
        self.all_weights = []
        self.particle = np.array([Particle(n_gaus=self.initialise_num(), noise_std_eff=self.noise_std_eff,
                                           num_evolution=self.num_evolution, mean_evolution=self.mean_evolution,
                                           std_evolution=self.std_evolution, amp_evolution=self.amp_evolution,
                                           noise_evolution=self.noise_evolution,
                                           prior_num=self.prior_num, prior_m=self.prior_m, prior_s=self.prior_s,
                                           prior_a=self.prior_a, prior_n=self.prior_n)
                                  for _ in range(0, self.n_particles)])
        for _p in self.particle:
            _p.weight = 1 / self.n_particles

        self.est_num = None
        self.all_noise_std = None
        self.n_iter = None
        self.est_num_avg = None
        self.noise_posterior = None
        self.particle_avg = []
        self.vector_mean = []
        self.vector_std = []
        self.vector_amp = []
        self.vector_noise_std = []
        self.vector_weight = []

    def initialise_num(self):
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
            mod_sel[self.particle[i].n_gaus] += self.particle[i].weight

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
                self.exponent_like = np.append(self.exponent_like, self.evolution_exponent())
                if n > 1 and self.mh_evolution:
                    post_aux = copy.deepcopy(self)
                    post_aux = post_aux.importance_sampling(post_aux.exponent_like[-1])
                    diff = abs(post_aux.norm_cost[-1] - self.norm_cost[-1]) / self.norm_cost[-1]
                else:
                    diff = 0
                    self = self.importance_sampling(self.exponent_like[-1])
            self = self.resampling()
            self = self.store_iteration()
            print(f'iter:{n} -- exp: {self.exponent_like[n]} -- est num: {np.argmax(self.mod_sel[n][:])}')
            if self.mh_evolution:
                self = copy.deepcopy(post_aux)
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
        integral_weight_u = []

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
            mod_sel_avg[self.particle_avg[i].n_gaus] += self.particle_avg[i].weight

        self.est_num_avg = np.argmax(mod_sel_avg)

        return self

    def vector_post(self):
        for _p in self.particle:
            for j in range(_p.n_gaus):
                self.vector_mean.append(_p.gaussian[j].mean)
                self.vector_std.append(_p.gaussian[j].std)
                self.vector_amp.append(_p.gaussian[j].amp)
                self.vector_noise_std.append(_p.noise_std)
                self.vector_weight.append(_p.weight / _p.n_gaus)

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
