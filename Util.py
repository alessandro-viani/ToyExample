# -*- coding: utf-8 -*-

# Author: Alessandro Viani <viani@dima.unige.it>
#
# License: BSD (3-clause)

import math as mat
import pickle

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def log_normal(x, mean, std):
    return -np.log(mat.sqrt(2 * np.pi) * std) - 0.5 * np.square((x - mean) / std)


def sequence_of_exponents(n_iter, max_exp):
    return np.concatenate((np.power(np.linspace(0, 1, n_iter), 4), [max_exp + 0.1]))


def creation_data(n_data=100, mean=[-2,2], std=[1, 0.5], amp=[0.7,0.3], 
                  min_soucespace=-5, max_sourcespace=5, noise_std=None,
                  linewidth=2, linestyle='-', color='k', alpha=0.6, dpi=1000, 
                  show_fig=False):
    sourcespace = np.linspace(min_soucespace, max_sourcespace, n_data)
    data = np.zeros(int(n_data))
    for i in range(n_data):
        data[i] = np.sum(amp * stats.norm.pdf(sourcespace[i], mean, std)) + np.random.normal(0, noise_std)
        
    plt.figure(figsize=(16, 9), dpi=100)
    x = np.linspace(-5, 5, 1000)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sum(amp * stats.norm.pdf(x[i], mean, std))
    plt.plot(sourcespace, data, '.', color='#1f77b4', markersize=7)
    plt.plot(x, y, linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha)
    plt.savefig('fig/data.png', format='png', dpi=dpi)
    if show_fig:
        plt.show()
    else:
        plt.close()    
    
    return sourcespace, data


def eval_kl_div(post_clas, post_prop):
    post_clas = post_clas / np.sum(post_clas)
    post_prop = post_prop / np.sum(post_prop)
    return sum(post_clas * np.log(np.divide(post_clas, post_prop)))


def bin_creation(interval, n_bins):
    left_bin = np.zeros(n_bins)
    right_bin = np.zeros(n_bins)
    center_bin = np.zeros(n_bins)

    for i in range(n_bins):
        left_bin[i] = interval[0] + i * np.abs(interval[-1] - interval[0]) / n_bins
        right_bin[i] = interval[0] + (i + 1) * np.abs(interval[-1] - interval[0]) / n_bins
        center_bin[i] = 0.5 * (left_bin[i] + right_bin[i])

    return left_bin, center_bin, right_bin


def exp_parameter(gaussian, weigth):
    exp_mean = 0
    exp_std = 0
    exp_amp = 0
    for idx, _g in enumerate(gaussian):
        exp_mean += _g.mean * weigth[idx]
        exp_std += _g.std * weigth[idx]
        exp_amp += _g.amp * weigth[idx]

    return exp_mean, exp_std, exp_amp


def err_parameters(particle, interval_mean, interval_std, interval_amp, n_bins, est_num, name_file=0):
    left_bin_mean, center_bin_mean, right_bin_mean = bin_creation(interval_mean, n_bins)
    left_bin_std, center_bin_std, right_bin_std = bin_creation(interval_std, n_bins)
    left_bin_amp, center_bin_amp, right_bin_amp = bin_creation(interval_amp, n_bins)

    if est_num == 1:
        single_gaussian = []
        weigth_single_gaussian = []
        for _p in particle:
            if _p.n_gaus == est_num:
                single_gaussian = np.append(single_gaussian, _p.gaussian[0])
                weigth_single_gaussian = np.append(weigth_single_gaussian, _p.weight)

        weigth_single_gaussian /= np.sum(weigth_single_gaussian)

        exp_mean, exp_std, exp_amp = exp_parameter(single_gaussian, weigth_single_gaussian)
        if exp_mean <= 0:
            exp_err_mean = np.abs(exp_mean - (-2))
            exp_err_std = np.abs(exp_std - 1)
            exp_err_amp = np.abs(exp_amp - 0.7)
        else:
            exp_err_mean = np.abs(exp_mean - 2)
            exp_err_std = np.abs(exp_std - 0.5)
            exp_err_amp = np.abs(exp_amp - 0.3)

        weight_bin_mean = np.zeros(n_bins)
        weight_bin_std = np.zeros(n_bins)
        weight_bin_amp = np.zeros(n_bins)

        for i in range(n_bins):
            for idx, _g in enumerate(single_gaussian):
                if left_bin_mean[i] <= _g.mean <= right_bin_mean[i]:
                    weight_bin_mean[i] += weigth_single_gaussian[idx]

                if left_bin_std[i] <= _g.std <= right_bin_std[i]:
                    weight_bin_std[i] += weigth_single_gaussian[idx]

                if left_bin_amp[i] <= _g.amp <= right_bin_amp[i]:
                    weight_bin_amp[i] += weigth_single_gaussian[idx]

        map_mean = center_bin_mean[np.argmax(weight_bin_mean)]
        map_std = center_bin_std[np.argmax(weight_bin_std)]
        map_amp = center_bin_amp[np.argmax(weight_bin_amp)]

        if map_mean <= 0:
            err_mean = np.abs(map_mean - (-2))
            err_std = np.abs(map_std - 1)
            err_amp = np.abs(map_amp - 0.7)

        else:
            err_mean = np.abs(map_mean - 2)
            err_std = np.abs(map_std - 0.5)
            err_amp = np.abs(map_amp - 0.3)

    if est_num == 2:
        first_gaussian = []
        second_gaussian = []

        weigth_two_gaussian_1 = []
        weigth_two_gaussian_2 = []

        for _p in particle:
            if _p.n_gaus == est_num:
                for i in range(int(est_num)):
                    if _p.gaussian[i].mean <= 0:
                        first_gaussian = np.append(first_gaussian, _p.gaussian[i])
                        weigth_two_gaussian_1 = np.append(weigth_two_gaussian_1, _p.weight)

                    elif _p.gaussian[i].mean > 0:
                        second_gaussian = np.append(second_gaussian, _p.gaussian[i])
                        weigth_two_gaussian_2 = np.append(weigth_two_gaussian_2, _p.weight)

        weigth_two_gaussian_1 /= np.sum(weigth_two_gaussian_1)
        weigth_two_gaussian_2 /= np.sum(weigth_two_gaussian_2)

        exp_mean_1, exp_std_1, exp_amp_1 = exp_parameter(first_gaussian, weigth_two_gaussian_1)
        exp_mean_2, exp_std_2, exp_amp_2 = exp_parameter(second_gaussian, weigth_two_gaussian_2)

        exp_err_mean = np.sum(np.abs(np.array([exp_mean_1, exp_mean_2]) - np.array([-2, 2])))
        exp_err_std = np.sum(np.abs(np.array([exp_std_1, exp_std_2]) - np.array([1, 0.5])))
        exp_err_amp = np.sum(np.abs(np.array([exp_amp_1, exp_amp_2]) - np.array([0.7, 0.3])))

        weight_bin_mean_1 = np.zeros(n_bins)
        weight_bin_mean_2 = np.zeros(n_bins)

        weight_bin_std_1 = np.zeros(n_bins)
        weight_bin_std_2 = np.zeros(n_bins)

        weight_bin_amp_1 = np.zeros(n_bins)
        weight_bin_amp_2 = np.zeros(n_bins)

        for i in range(n_bins):
            for idx, _g in enumerate(first_gaussian):
                if left_bin_mean[i] <= _g.mean <= right_bin_mean[i]:
                    weight_bin_mean_1[i] += weigth_two_gaussian_1[idx]

                if left_bin_std[i] <= _g.std <= right_bin_std[i]:
                    weight_bin_std_1[i] += weigth_two_gaussian_1[idx]

                if left_bin_amp[i] <= _g.amp <= right_bin_amp[i]:
                    weight_bin_amp_1[i] += weigth_two_gaussian_1[idx]

            for idx, _g in enumerate(second_gaussian):
                if left_bin_mean[i] <= _g.mean <= right_bin_mean[i]:
                    weight_bin_mean_2[i] += weigth_two_gaussian_2[idx]

                if left_bin_std[i] <= _g.std <= right_bin_std[i]:
                    weight_bin_std_2[i] += weigth_two_gaussian_2[idx]

                if left_bin_amp[i] <= _g.amp <= right_bin_amp[i]:
                    weight_bin_amp_2[i] += weigth_two_gaussian_2[idx]

        map_mean = np.array(
            [center_bin_mean[np.argmax(weight_bin_mean_1)], center_bin_mean[np.argmax(weight_bin_mean_2)]])
        map_std = np.array([center_bin_std[np.argmax(weight_bin_std_1)], center_bin_std[np.argmax(weight_bin_std_2)]])
        map_amp = np.array([center_bin_amp[np.argmax(weight_bin_amp_1)], center_bin_amp[np.argmax(weight_bin_amp_2)]])

        err_mean = np.sum(np.abs(map_mean - np.array([-2, 2])))
        err_std = np.sum(np.abs(map_std - np.array([1, 0.5])))
        err_amp = np.sum(np.abs(map_amp - np.array([0.7, 0.3])))

    return exp_err_mean, err_mean, exp_err_std, err_std, exp_err_amp, err_amp


def err_noise_std(particle, interval, n_bins, true_noise):
    left_bin, center_bin, right_bin = bin_creation(interval, n_bins)
    weight_bin = np.zeros(n_bins)

    cm_theta = 0
    for i in range(n_bins):
        for _p in particle:
            cm_theta += _p.weight * _p.noise_std
            if left_bin[i] <= _p.noise_std <= right_bin[i]:
                weight_bin[i] += _p.weight

    map_theta = center_bin[np.argmax(weight_bin)]

    return np.abs(map_theta - true_noise), np.abs(cm_theta - true_noise)


def analytics(n_file=1, interval_mean=[-5, 5], interval_std=[0.1, 10], interval_amp=[0.1, 10], n_bins=300,
              folder_name='save_folder/'):
    kl_div = np.zeros(n_file)

    var_clas = np.zeros(n_file)
    var_prop = np.zeros(n_file)

    est_n_clas = np.zeros(n_file)
    est_n_prop = np.zeros(n_file)

    map_m_clas = np.zeros(n_file)
    map_s_clas = np.zeros(n_file)
    map_a_clas = np.zeros(n_file)
    map_n_clas = np.zeros(n_file)

    map_m_prop = np.zeros(n_file)
    map_s_prop = np.zeros(n_file)
    map_a_prop = np.zeros(n_file)
    map_n_prop = np.zeros(n_file)

    cm_m_clas = np.zeros(n_file)
    cm_s_clas = np.zeros(n_file)
    cm_a_clas = np.zeros(n_file)
    cm_n_clas = np.zeros(n_file)

    cm_m_prop = np.zeros(n_file)
    cm_s_prop = np.zeros(n_file)
    cm_a_prop = np.zeros(n_file)
    cm_n_prop = np.zeros(n_file)

    for idx in range(n_file):
        with open(f'{folder_name}data_{idx}.pkl', 'rb') as f:
            sourcespace, data, _n = pickle.load(f)
        with open(f'{folder_name}posterior_proposed_{idx}.pkl', 'rb') as f:
            post_prop = pickle.load(f)
        with open(f'{folder_name}posterior_classical_{idx}.pkl', 'rb') as f:
            post_clas = pickle.load(f)

        var_prop[idx] = np.sum(post_prop.noise_posterior *
                               np.square(post_prop.all_noise_std - np.sum(
                                   post_prop.noise_posterior * post_prop.all_noise_std)))
        var_clas[idx] = np.sum(post_clas.vector_weight *
                               np.square(post_clas.vector_noise_std - np.sum(
                                   post_clas.vector_noise_std * post_clas.vector_weight)))
        est_n_prop[idx] = post_prop.est_num_avg
        cm_m_clas[idx], map_m_prop[idx], cm_s_prop[idx], \
        map_s_prop[idx], cm_a_prop[idx], map_a_prop[idx] = err_parameters(post_prop.particle_avg,
                                                                          interval_mean, interval_std,
                                                                          interval_amp, n_bins,
                                                                          post_prop.est_num_avg,
                                                                          name_file=f'_prop_{idx}')

        map_n_clas[idx] = np.abs(np.argmax(post_prop.noise_posterior) - _n)
        cm_n_clas[idx] = np.abs(np.sum(post_prop.noise_posterior * post_prop.all_noise_std) - _n)

        est_n_clas[idx] = post_clas.est_num
        cm_m_clas[idx], map_m_clas[idx], cm_s_clas[idx], \
        map_s_clas[idx], cm_a_clas[idx], map_a_clas[idx] = err_parameters(post_clas.particle,
                                                                          interval_mean, interval_std,
                                                                          interval_amp, n_bins,
                                                                          post_prop.est_num,
                                                                          name_file=f'_clas_{idx}')

        map_n_clas[idx], cm_n_clas[idx] = err_noise_std(post_clas.particle,
                                                        interval=[np.min(post_clas.vector_noise_std),
                                                                  np.max(post_clas.vector_noise_std)],
                                                        n_bins=100, true_noise=_n)

        kl_div[idx] = eval_kl_div(post_clas.noise_posterior, post_prop.noise_posterior)

    with open(f'{folder_name}analitics.pkl', 'wb') as f:
        pickle.dump([kl_div, var_clas, var_prop, est_n_clas, est_n_prop,
                     map_m_clas, map_s_clas, map_a_clas, map_n_clas,
                     map_m_prop, map_s_prop, map_a_prop, map_n_prop,
                     cm_m_clas, cm_s_clas, cm_a_clas, cm_n_clas,
                     cm_m_prop, cm_s_prop, cm_a_prop, cm_n_prop], f)
