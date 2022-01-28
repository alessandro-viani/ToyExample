import numpy as np
from Util import sequence_of_exponents
import scipy.stats as stats
from scipy.stats import poisson
from Util import log_normal
import copy 
from random import randint
from ClassParticle import ClassParticle

class ClassParameters(object):
    def __init__(self, num_evolution=None, mean_evolution=True, std_evolution=True,
                 amp_evolution=True, noise_evolution=False, sequence_evolution=True,
                 mh_evolution=False, sourcespace=None, data=None,
                max_exp=1, n_particles=100, max_num=3, noise_std_eff=None):

        self.num_evolution = num_evolution
        self.mean_evolution = mean_evolution
        self.std_evolution = std_evolution
        self.amp_evolution = amp_evolution

        self.noise_evolution = noise_evolution

        self.sequence_evolution = sequence_evolution
        self.mh_evolution = mh_evolution

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
        
        self.particle = np.array([ClassParticle(num=self.inizialize_num(), noise_std_eff=self.noise_std_eff, 
                                                num_evolution=self.num_evolution, 
                                 mean_evolution=self.mean_evolution, std_evolution=self.std_evolution,
                                 amp_evolution=self.amp_evolution, noise_evolution=self.noise_evolution) 
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
    
        self.ess = np.append(self.ess, 1/np.sum(np.power(weight, 2)))
        self.norm_cost = np.append(self.norm_cost, np.sum(weight_unnorm))
        
        return self
    
    def evolution_exponent(self):
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
            parameters_aux = copy.deepcopy(self)
            parameters_aux.ess[-1] = 0
            iterations = 1
            while not ess_min <= parameters_aux.ess[-1] / self.ess[-1] <= ess_max and iterations < 1e2:
                parameters_aux = copy.deepcopy(self)
                parameters_aux = parameters_aux.importance_sampling(next_exponent)
    
                if parameters_aux.ess[-1] / self.ess[-1] > ess_max:
                    delta_a = delta
                    delta = min((delta_a + delta_b) / 2, delta_max)
                    is_last_operation_increment = True
                    if delta_max - delta < delta_max / 100:
                        next_exponent = self.exponent_like[-1] + delta
                        parameters_aux = parameters_aux.importance_sampling(next_exponent)
                        if next_exponent >= 1:
                            next_exponent = 1
                            parameters_aux.ess[-1] = self.ess[-1] * (ess_max + ess_min) / 2
                        break
                else:
                    if parameters_aux.ess[-1] / self.ess[-1] < ess_min:
                        delta_b = delta
                        delta = max((delta_a + delta_b) / 2, delta_min)
                        if delta - delta_min < delta_min / 10 or \
                                (iterations > 1 and is_last_operation_increment == True):
                            next_exponent = self.exponent_like[-1] + delta
                            parameters_aux = parameters_aux.importance_sampling(next_exponent)
                            if next_exponent >= 1:
                                next_exponent = 1
                                parameters_aux.ess[-1] = self.ess[-1] * (ess_max + ess_min) / 2
                            break
                            is_last_operation_increment = False
                next_exponent = self.exponent_like[-1] + delta
                if next_exponent >= 1:
                    next_exponent = 1
                    parameters_aux.ess[-1] = self.ess[-1] * (ess_max + ess_min) / 2
                iterations += 1
    
        return next_exponent


