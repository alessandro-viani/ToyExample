import copy

import numpy as np
import scipy.stats as stats

from Util import log_normal


def mean_prior(x):
    prior = 0
    if -5 <= x <= 5:
        prior = 1 / 10
    return prior


def theta_prior(theta, theta_eff):
    return stats.gamma.pdf(theta, a=2, scale=4 * theta_eff)


def evaluation_likelihood(mean, theta, sourcespace, data, exponent_like):
    likelihood = 1
    if exponent_like > 0:
        log_likelihood = 0
        for idx, _d in enumerate(data):
            log_likelihood += log_normal(_d, np.exp(log_normal(sourcespace[idx], mean, 1)), theta)
        likelihood = np.exp(exponent_like * log_likelihood)
    return likelihood


def inizialize_mean():
    mean = -5 + 10 * np.random.rand()
    return mean


def inizialize_theta(theta_eff, method):
    theta = theta_eff
    if method == 'FB' or method == 'EM':
        theta = np.random.gamma(shape=2, scale=4 * theta_eff)
    return theta


class Particle(object):
    def __init__(self, cfg=None):

        if cfg is None:
            cfg = []

        if 'theta_eff' not in cfg:
            print('Error: set an estimate for the noise standard deviation')
        else:
            self.theta_eff = cfg['theta_eff']

        if 'method' not in cfg:
            print('Error: select one of the three methods "FB" - "EM" - "PM"')
        else:
            self.method = cfg['method']

        self.theta = inizialize_theta(self.theta_eff, self.method)
        self.mean = inizialize_mean()
        self.like = 1
        self.prior = mean_prior(self.mean)
        if self.method == 'FB' or self.method == 'EM':
            self.prior *= theta_prior(self.theta, self.theta_eff)
        self.weight = None
        self.weight_u = 1

    def mh_mean(self, sourcespace, data, exponent_like):
        part_aux = copy.deepcopy(self)

        part_aux.mean = np.random.normal(self.mean, 0.1)
        part_aux.like = evaluation_likelihood(part_aux.mean, part_aux.theta, sourcespace, data, exponent_like)
        part_aux.prior = mean_prior(part_aux.mean)

        rapp_prior = part_aux.prior / self.prior
        rapp_proposal = stats.norm.pdf(self.mean, part_aux.mean, 0.1) / stats.norm.pdf(part_aux.mean, self.mean, 0.1)
        rapp_like = part_aux.like / self.like

        if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
            self = copy.deepcopy(part_aux)

        return self

    def mh_theta(self, sourcespace, data, exponent_like):
        part_aux = copy.deepcopy(self)

        part_aux.theta = np.random.gamma(shape=100, scale=self.theta / 100)
        part_aux.like = evaluation_likelihood(part_aux.mean, part_aux.theta, sourcespace, data, exponent_like)
        part_aux.prior = theta_prior(part_aux.theta, part_aux.theta_eff)

        rapp_prior = part_aux.prior / self.prior
        rapp_proposal = stats.gamma.pdf(self.theta, a=100,
                                        scale=part_aux.theta / 100) / stats.gamma.pdf(part_aux.theta,
                                                                                      a=100, scale=self.theta / 100)
        rapp_like = part_aux.like / self.like

        if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
            self = copy.deepcopy(part_aux)

        return self
