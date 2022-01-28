import numpy as np
from Util import sequence_of_exponents
import scipy.stats as stats
from scipy.stats import poisson
from Util import log_normal
import copy 
from random import randint
from ClassGaussian import ClassGaussian

class ClassParticle(object):
    def __init__(self, num, noise_std_eff=None, num_evolution=None, 
                 mean_evolution=True, std_evolution=True,
                 amp_evolution=True, noise_evolution=False):
        self.num = num
        self.noise_std_eff = noise_std_eff
        
        self.num_evolution = num_evolution
        self.mean_evolution = mean_evolution
        self.std_evolution = std_evolution
        self.amp_evolution = amp_evolution
        self.noise_evolution = noise_evolution
        self.noise_std = self.inizialize_noise_std()
        
        self.gaussian = []
        for j in range(self.num):
            self.gaussian = np.append(self.gaussian, 
                                    ClassGaussian(self.inizialize_mean(), 
                                                  self.inizialize_std(), 
                                                  self.inizialize_amp()))
        self.like = 1
        self.weight = None
        self.weight_unnorm = 1
        self.prior = self.evaluation_prior()

        
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
            prior *= self.num_prior(self.num)
        if self.noise_evolution:
            prior *= self.noise_prior(self.noise_std)
    
        return prior
    
    def num_prior(self, x):
        return poisson.pmf(x, 0.25)
    
    def mean_prior(self, x):
        prior = 0
        if -5 <= x <= 5:
            prior = 1 / 10
        return prior
    
    def std_prior(self, x):
        prior = 0
        if 0.1 <= x <= 10:
            prior = 1 / (np.log10(10 / 0.1) * x)
        return prior
    
    def amp_prior(self, x):
        return stats.norm.pdf(x, 1, 0.25)
    
    def noise_prior(self, x):
        return stats.gamma.pdf(x, a=2, scale=self.noise_std_eff * 4)

    def evaluation_likelihood_mean(self, sourcespace):
        like_mean = 0
        for _g in self.gaussian:
            like_mean += _g.amp * np.exp(log_normal(sourcespace, _g.mean, _g.std))
        return like_mean
    
    def evaluation_likelihood_std(self):
        return self.noise_std
    
    def evaluation_likelihood(self, sourcespace, data, exponent_like):
        likelihood = 1
        if exponent_like > 0:
            log_likelihood = 0
            for idx, _d in enumerate(data):
                like_mean = self.evaluation_likelihood_mean(sourcespace[idx])
                like_std = self.evaluation_likelihood_std()
                log_likelihood += log_normal(_d, like_mean, like_std)
    
            likelihood = np.exp(exponent_like * log_likelihood)
    
        return likelihood

    def inizialize_mean(self):
        if self.mean_evolution:
            return -5 + (np.random.rand() * (5 - (-5)))
        else:
            return -2
    
    def inizialize_std(self):
        if not self.std_evolution:
            return 1
        else:
            return 10 ** (np.log10(0.1) + np.log10(10 / 0.1) * np.random.rand())
    
    def inizialize_amp(self):
        if not self.amp_evolution:
            return 1
        else:
            return 1 + 0.25 * np.random.normal(0, 1)
        
    def inizialize_noise_std(self):
        if self.noise_evolution:
            return np.random.gamma(shape=2, scale=self.noise_std_eff * 4)
        else:
            return self.noise_std_eff


    def jacobian_evaluation(self, std):
        jacobian = 1
        if self.mean_evolution:
            jacobian *= 10
        if self.std_evolution:
            jacobian *= np.log10(10 / 0.1) * np.log(10) * std
        if self.amp_evolution:
            jacobian *= 0.25
        return jacobian


    def mh_num(self, parameters):
        Q_death = 1 / 10
        Q_birth = 1 - Q_death
        proposal_particle = copy.deepcopy(self)
        birth_death = np.random.rand()
    
        if birth_death <= Q_birth and self.num < parameters.max_num:
            proposal_particle.gaussian = np.append(proposal_particle.gaussian, 
                                                   ClassGaussian(proposal_particle.inizialize_mean(), 
                                                                 proposal_particle.inizialize_std(), 
                                                                 proposal_particle.inizialize_amp()))
            
            jacobian = proposal_particle.jacobian_evaluation(proposal_particle.gaussian[proposal_particle.num].std)
            proposal_particle.num += 1
            
            
            proposal_particle.like = proposal_particle.evaluation_likelihood(parameters.sourcespace, parameters.data, parameters.exponent_like[-1])
            proposal_particle.prior = proposal_particle.evaluation_prior()
    
            rapp_prior = proposal_particle.prior / self.prior
            rapp_proposal = Q_death / Q_birth
    
        elif birth_death > 1 - Q_death and self.num > 0:
            guassian_to_die = randint(0, self.num - 1)
            proposal_particle.gaussian = np.delete(proposal_particle.gaussian, guassian_to_die)
                
            jacobian = 1 / self.jacobian_evaluation(self.gaussian[guassian_to_die].std)
            proposal_particle.num -= 1
            
            proposal_particle.like = proposal_particle.evaluation_likelihood(parameters.sourcespace, parameters.data, parameters.exponent_like[-1])
            proposal_particle.prior = proposal_particle.evaluation_prior()
            
            rapp_prior = proposal_particle.prior / self.prior
            rapp_proposal = Q_birth / Q_death
    
        if proposal_particle.num != self.num:
            rapp_like = proposal_particle.like / self.like
    
            if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal * jacobian, 1]):
                self = copy.deepcopy(proposal_particle)
        
        return self


    def mean_proposal_value(self, x):
        return np.random.normal(x, 0.5)
    
    def rapp_mean_proposal(self, x, x_prop):
        return stats.norm.pdf(x, x_prop, 0.5) / stats.norm.pdf(x_prop, x, 0.5)
    
    def mh_mean(self, parameters):
        proposal_particle = copy.deepcopy(self)
        for g_idx, _g in enumerate(proposal_particle.gaussian):
            _g.mean = self.mean_proposal_value(self.gaussian[g_idx].mean)
            
            proposal_particle.like = proposal_particle.evaluation_likelihood(parameters.sourcespace, parameters.data, parameters.exponent_like[-1])
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
    
    def mh_std(self, parameters):
        proposal_particle = copy.deepcopy(self)
        for i in range(proposal_particle.num):
            proposal_particle.gaussian[i].std = self.std_proposal_value(self.gaussian[i].std)
            
            proposal_particle.like = proposal_particle.evaluation_likelihood(parameters.sourcespace, parameters.data, parameters.exponent_like[-1])
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
    
    def mh_amp(self, parameters):
        proposal_particle = copy.deepcopy(self)
        for i in range(0, proposal_particle.num):
            proposal_particle.gaussian[i].amp = self.amp_proposal_value(self.gaussian[i].amp)
            
            proposal_particle.like = proposal_particle.evaluation_likelihood(parameters.sourcespace, parameters.data, parameters.exponent_like[-1])
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
    
    def mh_noise(self, parameters):
        proposal_particle = copy.deepcopy(self)
        proposal_particle.noise_std = self.noise_std_proposal_value(self.noise_std)
        
        proposal_particle.like = proposal_particle.evaluation_likelihood(parameters.sourcespace, parameters.data, parameters.exponent_like[-1])
        proposal_particle.prior = proposal_particle.evaluation_prior()
        
        rapp_prior = proposal_particle.prior / self.prior
        rapp_proposal = self.rapp_noise_std_proposal(self.noise_std, proposal_particle.noise_std)
        rapp_like = proposal_particle.like / self.like
    
        if np.random.rand() < min([rapp_prior * rapp_like * rapp_proposal, 1]):
            self = copy.deepcopy(proposal_particle)
    
        return self