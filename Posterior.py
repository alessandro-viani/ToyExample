# -*- coding: utf-8 -*-
"""
@author: Alessandro Viani (2022)
"""
import copy
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from Particle import Particle
from Util import sequence_of_exponents, bin_creation, log_normal

class Posterior(object):
    """Single Particle class for SMC samplers.
       Parameters
       ----------
       n_particles : :py:class:`int`
           The number of particles for performing SMC samplers.
       theta_eff : :py:class:`double`
           The estimated noise standard deviation.
       num_evolution : :py:class:`None or int`
           None: the number is one of the parameters to be estimated,
           int: it represents the fixed number of gaussian for each particle.
       prop_method : :py:class:`bool`
           True: the noise standard deviation is one of the parameters to be estimated,
           False: the noise standard deviation is fixed as the estimated noise standard deviation value and at the very
                  last iteration recycle scheme and noise posterior are estimated using the proposed method.
       sequence_evolution : :py:class:`None or int`
           None: exponent of the sequence are adaptively chosen,
           int: number of iterations of Importance sampling.
       sourcespace : :py:class:`np.array([double])`
           Grid of x-axes points where data are collected.
       data : :py:class:`np.array([double])`
           Noisy samples form the sum of weighted gaussian functions.
       max_exp : :py:class:`int`
           Maximum exponent reached during the SMC samplers iterations.
       max_num : :py:class:`int`
           Maximum number of gaussian allowed in each particle.
       prior_mean : :py:class:`np.array([double, double])`
           The interval prior parameters for the uniform prior on gaussian mean.
       prior_theta : :py:class:`np.array([double, double])`
           The shape and scale parameters prior parameters for the gamma prior on noise standard deviation.
        Attributes
        ----------
       ess : :py:class:`np.array([double])`
           The noise standard deviation sampled if prop_method is True,
           or the noise standard deviation estimated if prop_method is False.
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

    def __init__(self, n_particles=None, theta_eff=None, prop_method=False, sequence_evolution=None,
                 sourcespace=None, data=None, prior_mean=[-5, 5], prior_theta=[2, 4],
                 max_exp=1, ess_min=0.99, ess_max=0.999, delta_min=1e-3, delta_max=1e-1,
                 point_spline=1e4, n_bins=10, verbose=True):

        self.n_particles = int(n_particles)
        self.theta_eff = theta_eff
        self.prop_method = prop_method
        self.sequence_evolution = sequence_evolution

        self.sourcespace = sourcespace
        self.data = data
        self.prior_mean = prior_mean
        self.prior_theta = prior_theta

        self.max_exp = max_exp
        self.ess_max = ess_max
        self.ess_min = ess_min
        self.delta_min = delta_min
        self.delta_max = delta_max

        self.point_spline = point_spline
        self.n_bins = n_bins
        self.verbose = verbose

        self.exponent_like = np.array([0, 0])
        self.ess = self.n_particles
        self.norm_cost = 1
        self.mod_sel = []
        self.particle = np.array([Particle(theta_eff=self.theta_eff,
                                           prop_method=self.prop_method,
                                           prior_mean=self.prior_mean,
                                           prior_theta=self.prior_theta)
                                  for _ in range(0, self.n_particles)])

        for _p in self.particle:
            _p.weight = 1 / self.n_particles

        self.all_particles = np.array([self.particle])
        self.all_weights_unnorm = np.array([np.ones(self.n_particles)])
        self.all_weights = 1/self.n_particles * np.array([np.ones(self.n_particles)])

        self.n_iter = None

        self.all_theta = None
        self.grid_theta = None
        self.theta_prior = None
        self.theta_posterior = None
        self.theta_likelihood = None
        self.theta_posterior_int = None
        self.theta_likelihood_int = None

        self.ml_theta = None
        self.map_theta = None
        self.pm_theta = None
        self.map_mean = None
        self.pm_mean = None

        self.particle_avg = []
        self.vector_mean = []
        self.vector_theta = []
        self.vector_weight = []
        self.cpu_time = None

    def metropolis_hastings(self):
        for idx, _p in enumerate(self.particle):
            self.particle[idx] = self.particle[idx].mh_mean(self)
            if not self.prop_method:
                self.particle[idx] = self.particle[idx].mh_theta(self)
        return self

    def importance_sampling(self, next_alpha):
        weight_unnorm = np.zeros(self.n_particles)
        for idx, _p in enumerate(self.particle):
            new_like = _p.evaluation_likelihood(self.sourcespace, self.data, next_alpha)
            if _p.like == 0:
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
        self.norm_cost = np.append(self.norm_cost, 1/self.n_particles * np.sum(weight_unnorm))

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

    def evolution_exponent(self):
        if self.sequence_evolution is None:
            if self.exponent_like[-1] == self.max_exp:
                next_exponent = self.max_exp + 0.1
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
                                    (iterations > 1 and is_last_operation_increment == True):
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
            next_exponent = sequence_of_exponents(self.sequence_evolution,
                                                  self.max_exp)[len(self.exponent_like)]

        return next_exponent

    def perform_smc(self):
        start_time = time.time()
        n = 0
        if self.verbose:
            print(f'iter:{n} -- exp: {self.exponent_like[n]}')

        self = self.importance_sampling(self.exponent_like[-1])

        self.all_particles = np.concatenate([self.all_particles, np.array([self.particle])], axis=0)
        self.all_weights_unnorm = np.concatenate(
            [self.all_weights_unnorm, np.array([[_p.weight_unnorm for _p in self.particle]])], axis=0)
        self.all_weights = np.concatenate([self.all_weights, np.array([[_p.weight for _p in self.particle]])])

        n = 1
        while self.exponent_like[-1] <= 1:
            self = self.metropolis_hastings()
            self.exponent_like = np.append(self.exponent_like, self.evolution_exponent())
            self = self.importance_sampling(self.exponent_like[-1])
            self = self.resampling()
            self = self.store_iteration()
            self.vector_post()
            self.mean_estimates(self.particle)
            if self.verbose:
                print(f'iter:{n} -- exp: {"{:.4f}".format(self.exponent_like[n])}')
            n += 1
        self.n_iter = n
        self.vector_post()

        if self.prop_method:
            self = self.compute_big_posterior()
            self.mean_estimates(self.particle_avg)
            self.theta_estimates()

            self = self.empirical_bayes()
        else:
            self.mean_estimates(self.particle)
            self.theta_estimates()

        self.cpu_time = time.time() - start_time
        if self.verbose:
            print('\n-- time for execution: %s (s) --' % self.cpu_time)

        return self

    def compute_big_posterior(self):
        integral_weight_u = []
        norm_cost = self.norm_cost[2: self.n_iter]
        exponent_like = self.exponent_like[2: self.n_iter]

        self.all_theta = self.theta_eff / np.sqrt(exponent_like)

        self.theta_prior = Particle(theta_eff=self.theta_eff,
                     prop_method=self.prop_method,
                     prior_mean=self.prior_mean,
                     prior_theta=self.prior_theta).theta_prior(self.all_theta)

        delta_std = np.zeros(len(self.all_theta))
        delta_std[0] = abs(self.all_theta[0] - self.all_theta[1])
        delta_std[-1] = abs(self.all_theta[-2] - self.all_theta[-1])
        for i in range(2, len(self.all_theta)):
            delta_std[i - 1] = abs(self.all_theta[i - 2] - self.all_theta[i])

        k = np.power(np.power(2 * np.pi * np.square(self.theta_eff),
                              exponent_like - 1) * exponent_like,
                     self.data.shape[0] / 2)

        weight_upgrade = 0.5 * delta_std * k * self.theta_prior * norm_cost

        for t_idx in range(self.n_iter-2):
            for p_idx in range(self.n_particles):
                integral_weight_u = np.append(integral_weight_u, self.all_weights[t_idx + 2:, p_idx] * weight_upgrade[t_idx])
                p_aux = copy.deepcopy(self.all_particles[t_idx + 2, p_idx])
                self.particle_avg = np.append(self.particle_avg, p_aux)

        integral_weight = integral_weight_u / np.sum(integral_weight_u)

        for idx, _p in enumerate(self.particle_avg):
            _p.weight_unnorm = integral_weight_u[idx]
            _p.weight = integral_weight[idx]

        self.theta_likelihood = norm_cost * k
        self.theta_posterior = self.theta_prior * self.theta_likelihood / np.sum(self.theta_prior * self.theta_likelihood)

        return self

    def mean_estimates(self, particle):
        self.pm_mean = 0
        for _p in particle:
            self.pm_mean += _p.gaussian.mean * _p.weight

        left_bin, center_bin, right_bin = bin_creation(np.array([np.min(self.vector_mean),
                                                                 np.max(self.vector_mean)]),
                                                       self.n_bins)
        weight_bin = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            for idx, _p in enumerate(particle):
                if left_bin[i] <= _p.gaussian.mean <= right_bin[i]:
                    weight_bin[i] += _p.weight
        self.map_mean = center_bin[np.argmax(weight_bin)]


    def theta_estimates(self):
        if self.prop_method:

            integral = 0
            self.pm_theta = 0
            self.grid_theta = np.unique(
                                np.sort(
                                    np.append(self.all_theta,
                                              np.linspace(np.min(self.all_theta),
                                                          np.max(self.all_theta),
                                                          int(self.point_spline)))))
            # self.theta_posterior_int = scipy.interpolate.splev(self.grid_theta,
            #                                                       scipy.interpolate.splrep(self.all_theta[::-1],
            #                                                                                self.theta_posterior[::-1]))
            self.theta_posterior_int = scipy.interpolate.interp1d(self.all_theta,
                                                                  self.theta_posterior,
                                                                  kind='linear')(self.grid_theta)
            # self.theta_likelihood_int = scipy.interpolate.splev(self.grid_theta,
            #                                                         scipy.interpolate.splrep(self.all_theta[::-1],
            #                                                                                self.theta_likelihood[::-1]))
            self.theta_likelihood_int = scipy.interpolate.interp1d(self.all_theta,
                                                                  self.theta_likelihood,
                                                                  kind='linear')(self.grid_theta)

            delta = np.abs(self.grid_theta[:-1] - self.grid_theta[1:])

            integral = 0.5 * np.sum((self.theta_posterior_int[:-1] + self.theta_posterior_int[1:]) * delta)


            self.pm_theta = 0.5 * np.sum((self.grid_theta[:-1] * self.theta_posterior_int[:-1] +
                                         self.grid_theta[1:] * self.theta_posterior_int[1:]) * delta)

            #conditional mean
            self.pm_theta /= integral
            #maximum a posteriori
            self.map_theta = self.grid_theta[np.argmax(self.theta_posterior_int)]
            #maximum likelihood
            self.ml_theta = self.grid_theta[np.argmax(self.theta_likelihood_int)]

        else:
            left_bin_theta, center_bin_theta, right_bin_theta = bin_creation(np.array([np.min(self.vector_theta),
                                                                                       np.max(self.vector_theta)]),
                                                                             self.n_bins)
            weight_bin_theta = np.zeros(self.n_bins)
            self.pm_theta = np.sum(np.array(self.vector_theta) * np.array(self.vector_weight))

            for idx in range(self.n_bins):
                for jdx, _n in enumerate(self.vector_theta):
                    if left_bin_theta[idx] <= _n <= right_bin_theta[idx]:
                        weight_bin_theta[idx] += self.vector_weight[jdx]
            self.map_theta = center_bin_theta[np.argmax(weight_bin_theta)]


    def empirical_bayes(self):
        aux_alpha = np.power(self.theta_eff/self.ml_theta, 2)

        self.idx_max = 0
        while self.exponent_like[self.idx_max] < aux_alpha and self.idx_max<=self.n_iter+1:
            self.idx_max +=1
        self.idx_max -=1

        posterior_eb = Posterior(prop_method=True,
                    sourcespace=self.sourcespace,
                    data=self.data,
                    n_particles=self.n_particles,
                    theta_eff=self.theta_eff,
                    prior_mean=self.prior_mean,
                    prior_theta=self.prior_theta)

        for idx, _p in enumerate(self.all_particles[self.idx_max]):
            posterior_eb.particle[idx] = copy.deepcopy(_p)

        posterior_eb = posterior_eb.importance_sampling(next_alpha=aux_alpha)

        posterior_eb.pm_smc = self.pm_theta
        posterior_eb.map_smc = self.map_theta

        posterior_eb.grid_theta = self.grid_theta
        posterior_eb.all_theta = self.all_theta
        posterior_eb.theta_posterior = self.theta_posterior
        posterior_eb.theta_posterior_int = self.theta_posterior_int
        posterior_eb.ml_theta = self.ml_theta
        posterior_eb.map_theta = self.map_theta
        posterior_eb.particle_avg = posterior_eb.particle
        posterior_eb.vector_post()

        self.posterior_eb = posterior_eb

        return self

    def vector_post(self):
        self.vector_mean = []
        self.vector_theta = []
        self.vector_weight = []
        for _p in self.particle:
            self.vector_mean.append(_p.gaussian.mean)
            self.vector_theta.append(_p.theta)
            self.vector_weight.append(_p.weight)

    def store_iteration(self):
        self.all_particles = np.concatenate([self.all_particles, np.array([self.particle])], axis=0)
        self.all_weights_unnorm = np.concatenate([self.all_weights_unnorm,
                                                  np.array([[_p.weight_unnorm for _p in self.particle]])], axis=0)
        self.all_weights = np.concatenate([self.all_weights, np.array([[_p.weight for _p in self.particle]])])

        return self

    def plot_marginals(self, linewidth=1, alpha=0.5, dpi=300, plot_show=False):

        color_map = ['#1f77b4', 'darkorange', 'forestgreen', 'red']
        color_hist = 0
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        x = np.linspace(-5, 5, 1000)
        y = 1 * scipy.stats.norm.pdf(x, 0, 1)
        ax[0].plot(self.sourcespace, self.data, '.', color='#1f77b4', markersize=7)
        ax[0].plot(x, y, linestyle='-', color='k', linewidth=linewidth, alpha=alpha)
        ax[0].set_title('Data')
        ax[0].set_xlabel(r'$x$')
        ax[0].set_ylabel(r'$y$')

        sns.histplot(x=self.vector_mean, stat='probability',
                     weights=self.vector_weight, bins=self.n_bins,
                     color=color_map[color_hist], alpha=alpha, ax=ax[1])
        ax[1].set_xlabel(r'$\mu$')
        ax[1].set_title(r'$p(\mu\mid y)$')
        # ax[1].set_xlim(self.prior_mean)
        ax[1].set_ylabel('', fontsize=0)

        if self.prop_method:
            ax[2].plot(self.all_theta, self.theta_posterior, '.',
                       color=color_map[color_hist])
            ax[2].fill_between(self.grid_theta, self.theta_posterior_int,
                               color=color_map[color_hist], alpha=alpha)
            ax[2].plot(self.grid_theta, self.theta_posterior_int, '-')
            ax[2].set_xlim([np.min(self.all_theta), np.max(self.all_theta)])
            plt.sca(ax[2])
        else:
            sns.set_style('darkgrid')
            sns.histplot(x=self.vector_theta, stat='probability',
                         weights=self.vector_weight, bins=self.n_bins,
                         color=color_map[color_hist], alpha=alpha, ax=ax[2])
            ax[2].set_ylabel('', fontsize=0)

        ax[2].set_xlabel(r'$\theta$')
        ax[2].set_title(r'$p(\theta\mid y)$')
        fig.tight_layout()
        if self.prop_method:
            fig.savefig('fig/plot_marginals_prop_method.png', format='png', dpi=dpi)
        else:
            fig.savefig('fig/plot_marginals_fully_bayesian.png', format='png', dpi=dpi)
        if plot_show:
            plt.show()
        else:
            plt.close()
