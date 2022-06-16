# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:48:19 2022

@author: viani
"""

import numpy as np
import scipy.stats as stats


def log_normal(x, mean, std):
    return -np.log(np.sqrt(2 * np.pi) * std) - 0.5 * np.square((x - mean) / std)


def sequence_of_exponents(n_iter, max_exp):
    return np.concatenate((np.power(np.linspace(0, 1, n_iter), 4), [max_exp + 0.1]))


def creation_data(n_data, noise_std):
    sourcespace = np.linspace(-5, 5, n_data)
    data = np.zeros(int(n_data))
    for i in range(0, n_data):
        data[i] = 1 * stats.norm.pdf(sourcespace[i], 0, 1) + np.random.normal(0, noise_std)
    return sourcespace, data


def bin_creation(interval, n_bins):
    left_bin = np.zeros(n_bins)
    right_bin = np.zeros(n_bins)
    center_bin = np.zeros(n_bins)

    for i in range(n_bins):
        left_bin[i] = interval[0] + i * np.abs(interval[-1] - interval[0]) / n_bins
        right_bin[i] = interval[0] + (i + 1) * np.abs(interval[-1] - interval[0]) / n_bins
        center_bin[i] = 0.5 * (left_bin[i] + right_bin[i])

    return left_bin, center_bin, right_bin
