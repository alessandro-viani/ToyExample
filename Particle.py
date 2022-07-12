# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:48:51 2022

@author: viani
"""
import copy
import numpy as np
import scipy.stats as stats
from Gaussian import Gaussian
from Util import log_normal

class Particle(object):
    def __init__(self,
                 theta_eff=None,
                 prop_method=False,
                 prior_mean=None,
                 prior_theta=None):
        self.theta_eff = theta_eff
        self.prop_method = prop_method

        self.prior_mean = prior_mean
        self.prior_theta = prior_theta

        self.theta = self.inizialize_theta()
        self.gaussian = Gaussian(self.inizialize_mean(), 1, 1)
        self.like = 1
        self.prior = self.evaluation_prior()
        self.weight = None
        self.weight_unnorm = 1

    def evaluation_prior(self):
        prior = self.mean_prior(self.gaussian.mean)
        if not self.prop_method:
            prior *= self.theta_prior(self.theta)

        return prior

    def mean_prior(self, x):
        prior = 0
        if self.prior_mean[0] <= x <= self.prior_mean[1]:
            prior = 1 / np.abs(self.prior_mean[0] - self.prior_mean[1])
        return prior

    def theta_prior(self, x):
        return stats.gamma.pdf(x, a=self.prior_theta[0], scale=self.theta_eff * self.prior_theta[1])

    def evaluation_likelihood(self, sourcespace, data, exponent_like):
        likelihood = 1
        if exponent_like > 0:
            log_likelihood = 0
            for idx, _d in enumerate(data):
                log_likelihood += log_normal(_d,
                                             self.gaussian.amp * np.exp(log_normal(sourcespace[idx], self.gaussian.mean, self.gaussian.std)),
                                             self.theta)

            likelihood = np.exp(exponent_like * log_likelihood)

        return likelihood

    def inizialize_mean(self):
        mean = self.prior_mean[0] + (np.random.rand() * np.abs(self.prior_mean[1] - self.prior_mean[0]))
        return mean

    def mean_proposal_value(self, x):
        return np.random.normal(x, 0.1)

    def rapp_mean_proposal(self, x, x_prop):
        return stats.norm.pdf(x, x_prop, 0.1) / stats.norm.pdf(x_prop, x, 0.1)

    def mh_mean(self, post):
        proposal_particle = copy.deepcopy(self)
        proposal_particle.gaussian.mean = self.mean_proposal_value(self.gaussian.mean)

        proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                         post.exponent_like[-1])
        proposal_particle.prior = proposal_particle.evaluation_prior()

        rapp_prior = proposal_particle.prior / self.prior
        rapp_proposal = self.rapp_mean_proposal(self.gaussian.mean, proposal_particle.gaussian.mean)
        rapp_like = proposal_particle.like / self.like

        if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
            self = copy.deepcopy(proposal_particle)

        return self

    def inizialize_theta(self):
        theta = self.theta_eff
        if not self.prop_method:
            theta = np.random.gamma(shape=self.prior_theta[0], scale=self.theta_eff * self.prior_theta[1])
        return theta

    def theta_proposal_value(self, actual_theta):
        return np.random.gamma(shape=100, scale=actual_theta / 100)

    def rapp_theta_proposal(self, x, x_prop):
        return stats.gamma.pdf(x, a=100, scale=x_prop / 100) / stats.gamma.pdf(x_prop, a=100, scale=x / 100)

    def mh_noise(self, post):
        proposal_particle = copy.deepcopy(self)
        proposal_particle.theta = self.theta_proposal_value(self.theta)

        proposal_particle.like = proposal_particle.evaluation_likelihood(post.sourcespace, post.data,
                                                                         post.exponent_like[-1])
        proposal_particle.prior = proposal_particle.evaluation_prior()

        rapp_prior = proposal_particle.prior / self.prior
        rapp_proposal = self.rapp_theta_proposal(self.theta, proposal_particle.theta)
        rapp_like = proposal_particle.like / self.like

        if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
            self = copy.deepcopy(proposal_particle)

        return self
