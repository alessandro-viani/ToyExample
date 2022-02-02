# -*- coding: utf-8 -*-

# Author: Alessandro Viani <viani@dima.unige.it>
#
# License: BSD (3-clause)

import copy
from random import randint

import numpy as np
import scipy.stats as stats

from Gaussian import Gaussian
from Util import log_normal


class Particle(object):
    """Single Particle class for SMC samplers.

       Parameters
       ----------
       n_gaus : :py:class:`int`
           The number of gaussian in the particle.
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
       prop_method : :py:class:`bool`
           True: the noise standard deviation is one of the parameters to be estimated,
           False: the noise standard deviation is fixed as the estimated noise standard deviation value and at the very
                  last iteration recycle scheme and noise posterior are estimated using the proposed method.
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
       noise_std : :py:class:`double`
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

    def __init__(self, n_gaus=None, noise_std_eff=None,
                 num_evolution=None, mean_evolution=True, std_evolution=True, amp_evolution=True, prop_method=False,
                 prior_num=None, prior_m=None, prior_s=None, prior_a=None, prior_n=None):
        self.n_gaus = n_gaus
        self.noise_std_eff = noise_std_eff

        self.num_evolution = num_evolution
        self.mean_evolution = mean_evolution
        self.std_evolution = std_evolution
        self.amp_evolution = amp_evolution
        self.prop_method = prop_method

        self.prior_num = prior_num
        self.prior_m = prior_m
        self.prior_s = prior_s
        self.prior_a = prior_a
        self.prior_n = prior_n

        self.noise_std = self.inizialize_noise_std()
        self.q_death = 1 / 10
        self.q_birth = 1 - self.q_death
        self.gaussian = []
        for _ in range(self.n_gaus):
            self.gaussian = np.append(self.gaussian, Gaussian(self.inizialize_mean(),
                                                              self.inizialize_std(),
                                                              self.inizialize_amp()))
        self.like = 1
        self.prior = self.evaluation_prior()
        self.weight = None
        self.weight_unnorm = 1

    def evaluation_prior(self):
        prior = 1
        for _g in self.gaussian:
            if self.mean_evolution:
                prior *= self.mean_prior(_g.mean)
            if self.std_evolution:
                prior *= self.std_prior(_g.std)
            if self.amp_evolution:
                prior *= self.amp_prior(_g.amp)
        if self.num_evolution is None:
            prior *= self.num_prior(self.n_gaus)
        if self.prop_method:
            prior *= self.noise_prior(self.noise_std)

        return prior

    def num_prior(self, x):
        return stats.poisson.pmf(x, self.prior_num)

    def mean_prior(self, x):
        prior = 0
        if self.prior_m[0] <= x <= self.prior_m[1]:
            prior = 1 / np.abs(self.prior_m[0] - self.prior_m[1])
        return prior

    def std_prior(self, x):
        prior = 0
        if self.prior_s[0] <= x <= self.prior_s[1]:
            prior = 1 / (np.log10(self.prior_s[1] / self.prior_s[0]) * x)
        return prior

    def amp_prior(self, x):
        return stats.norm.pdf(x, self.prior_a[0], self.prior_a[1])

    def noise_prior(self, x):
        return stats.gamma.pdf(x, a=self.prior_n[0], scale=self.noise_std_eff * self.prior_n[1])

    def evaluation_likelihood(self, sourcespace, data, exponent_like):
        likelihood = 1
        if exponent_like > 0:
            log_likelihood = 0
            for idx, _d in enumerate(data):
                like_mean = 0
                for _g in self.gaussian:
                    like_mean += _g.amp * np.exp(log_normal(sourcespace[idx], _g.mean, _g.std))
                log_likelihood += log_normal(_d, like_mean, self.noise_std)

            likelihood = np.exp(exponent_like * log_likelihood)

        return likelihood

    def inizialize_mean(self):
        if self.mean_evolution:
            return self.prior_m[0] + (np.random.rand() * np.abs(self.prior_m[1] - self.prior_m[0]))
        else:
            return -2

    def inizialize_std(self):
        if not self.std_evolution:
            return 1
        else:
            return self.prior_s[1] ** (
                    np.log10(self.prior_s[0]) + np.log10(self.prior_s[1] / self.prior_s[0]) * np.random.rand())

    def inizialize_amp(self):
        if not self.amp_evolution:
            return 1
        else:
            return self.prior_a[0] + self.prior_a[1] * np.random.normal(0, 1)

    def inizialize_noise_std(self):
        if self.prop_method:
            return np.random.gamma(shape=self.prior_n[0], scale=self.noise_std_eff * self.prior_n[1])
        else:
            return self.noise_std_eff

    def jacobian_evaluation(self, std):
        jacobian = 1
        if self.mean_evolution:
            jacobian *= np.abs(self.prior_m[1] - self.prior_m[0])
        if self.std_evolution:
            jacobian *= np.log10(self.prior_s[1] / self.prior_s[0]) * np.log(self.prior_s[1]) * std
        if self.amp_evolution:
            jacobian *= self.prior_a[1]
        return jacobian

    def mh_num(self, post):
        proposal_particle = copy.deepcopy(self)
        birth_death = np.random.rand()

        if birth_death <= self.q_birth and self.n_gaus < post.max_num:
            proposal_particle.gaussian = np.append(proposal_particle.gaussian,
                                                   Gaussian(proposal_particle.inizialize_mean(),
                                                            proposal_particle.inizialize_std(),
                                                            proposal_particle.inizialize_amp()))

            jacobian = proposal_particle.jacobian_evaluation(proposal_particle.gaussian[proposal_particle.n_gaus].std)
            proposal_particle.n_gaus += 1

            proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                             post.exponent_like[-1])
            proposal_particle.prior = proposal_particle.evaluation_prior()

            rapp_prior = proposal_particle.prior / self.prior
            rapp_proposal = self.q_death / self.q_birth

        elif birth_death > 1 - self.q_death and self.n_gaus > 0:
            guassian_to_die = randint(0, self.n_gaus - 1)
            proposal_particle.gaussian = np.delete(proposal_particle.gaussian, guassian_to_die)

            jacobian = 1 / self.jacobian_evaluation(self.gaussian[guassian_to_die].std)
            proposal_particle.n_gaus -= 1

            proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                             post.exponent_like[-1])
            proposal_particle.prior = proposal_particle.evaluation_prior()

            rapp_prior = proposal_particle.prior / self.prior
            rapp_proposal = self.q_birth / self.q_death

        if proposal_particle.n_gaus != self.n_gaus:
            rapp_like = proposal_particle.like / self.like

            if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal * jacobian, 1]):
                self = copy.deepcopy(proposal_particle)

        return self

    def mean_proposal_value(self, x):
        return np.random.normal(x, 0.5)

    def rapp_mean_proposal(self, x, x_prop):
        return stats.norm.pdf(x, x_prop, 0.5) / stats.norm.pdf(x_prop, x, 0.5)

    def mh_mean(self, post):
        proposal_particle = copy.deepcopy(self)
        for g_idx, _g in enumerate(proposal_particle.gaussian):
            _g.mean = self.mean_proposal_value(self.gaussian[g_idx].mean)

            proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                             post.exponent_like[-1])
            proposal_particle.prior = proposal_particle.evaluation_prior()

            rapp_prior = proposal_particle.prior / self.prior
            rapp_proposal = self.rapp_mean_proposal(self.gaussian[g_idx].mean, _g.mean)
            rapp_like = proposal_particle.like / self.like

            if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
                self = copy.deepcopy(proposal_particle)

        return self

    def std_proposal_value(self, x):
        return np.random.gamma(shape=100, scale=x / 100)

    def rapp_std_proposal(self, x, x_prop):
        return stats.gamma.pdf(x, a=100, scale=x_prop / 100) / stats.gamma.pdf(x_prop, a=100, scale=x / 100)

    def mh_std(self, post):
        proposal_particle = copy.deepcopy(self)
        for i in range(proposal_particle.n_gaus):
            proposal_particle.gaussian[i].std = self.std_proposal_value(self.gaussian[i].std)

            proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                             post.exponent_like[-1])
            proposal_particle.prior = proposal_particle.evaluation_prior()

            rapp_prior = proposal_particle.prior / self.prior
            rapp_proposal = self.rapp_std_proposal(self.gaussian[i].std, proposal_particle.gaussian[i].std)
            rapp_like = proposal_particle.like / self.like

            if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
                self = copy.deepcopy(proposal_particle)

        return self

    def amp_proposal_value(self, actual_amp):
        return np.random.gamma(shape=100, scale=actual_amp / 100)

    def rapp_amp_proposal(self, x, x_prop):
        return stats.gamma.pdf(x, a=100, scale=x_prop / 100) / stats.gamma.pdf(x_prop, a=100, scale=x / 100)

    def mh_amp(self, post):
        proposal_particle = copy.deepcopy(self)
        for i in range(0, proposal_particle.n_gaus):
            proposal_particle.gaussian[i].amp = self.amp_proposal_value(self.gaussian[i].amp)

            proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                             post.exponent_like[-1])
            proposal_particle.prior = proposal_particle.evaluation_prior()

            rapp_prior = proposal_particle.prior / self.prior
            rapp_proposal = self.rapp_amp_proposal(self.gaussian[i].amp, proposal_particle.gaussian[i].amp)
            rapp_like = proposal_particle.like / self.like

            if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
                self = copy.deepcopy(proposal_particle)

        return self

    def noise_std_proposal_value(self, actual_noise_std):
        return np.random.gamma(shape=100, scale=actual_noise_std / 100)

    def rapp_noise_std_proposal(self, x, x_prop):
        return stats.gamma.pdf(x, a=100, scale=x_prop / 100) / stats.gamma.pdf(x_prop, a=100, scale=x / 100)

    def mh_noise(self, post):
        proposal_particle = copy.deepcopy(self)
        proposal_particle.noise_std = self.noise_std_proposal_value(self.noise_std)

        proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                         post.exponent_like[-1])
        proposal_particle.prior = proposal_particle.evaluation_prior()

        rapp_prior = proposal_particle.prior / self.prior
        rapp_proposal = self.rapp_noise_std_proposal(self.noise_std, proposal_particle.noise_std)
        rapp_like = proposal_particle.like / self.like

        if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
            self = copy.deepcopy(proposal_particle)

        return self
