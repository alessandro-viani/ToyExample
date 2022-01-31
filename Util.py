import math as mat
import numpy as np
import scipy.stats as stats
from numba import jit



def evaluation_likelihood_slow(part, sourcespace, data, exponent_like):
    likelihood = 1
    if exponent_like > 0:
        log_likelihood = 0
        for idx, _d in enumerate(data):
            like_mean = 0
            for _g in part.gaussian:
                like_mean += _g.amp * np.exp(log_normal(sourcespace[idx], _g.mean, _g.std))
            log_likelihood += log_normal(_d, like_mean, part.noise_std)

        likelihood = np.exp(exponent_like * log_likelihood)

    return likelihood

def evaluation_likelihood(part, sourcespace, data, exponent_like):
    return jit(part, sourcespace, data, exponent_like)(evaluation_likelihood_slow)

def log_normal(x, mean, std):
    return -np.log(mat.sqrt(2 * np.pi) * std) - 0.5 * np.square((x - mean) / std)


def sequence_of_exponents(n_iter, max_exp):
    return np.concatenate((np.power(np.linspace(0, 1, n_iter), 4), [max_exp + 0.1]))


def creation_data(n_data, noise_std):
    sourcespace = np.linspace(-5, 5, n_data)
    data = np.zeros(int(n_data))
    for i in range(0, n_data):
        data[i] = 0.7 * stats.norm.pdf(sourcespace[i], -2, 1) + \
                  0.3 * stats.norm.pdf(sourcespace[i], 2, 0.5) + \
                  np.random.normal(0, noise_std)
    return sourcespace, data


def eval_kl_div(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return sum(p * np.log(np.divide(p, q)))


def error_parameters(particle, interval_mean, interval_std, interval_amp, n_bins, est_num, name_file=0):
    len_interval_mean = np.abs(interval_mean[-1] - interval_mean[0])
    left_bin_mean = np.zeros(n_bins)
    rigtht_bin_mean = np.zeros(n_bins)
    center_bin_mean = np.zeros(n_bins)

    len_interval_std = np.abs(interval_std[-1] - interval_std[0])
    left_bin_std = np.zeros(n_bins)
    rigtht_bin_std = np.zeros(n_bins)
    center_bin_std = np.zeros(n_bins)

    len_interval_amp = np.abs(interval_amp[-1] - interval_amp[0])
    left_bin_amp = np.zeros(n_bins)
    rigtht_bin_amp = np.zeros(n_bins)
    center_bin_amp = np.zeros(n_bins)

    for i in range(n_bins):
        left_bin_mean[i] = interval_mean[0] + i * len_interval_mean / n_bins
        rigtht_bin_mean[i] = interval_mean[0] + (i + 1) * len_interval_mean / n_bins
        center_bin_mean[i] = 0.5 * (left_bin_mean[i] + rigtht_bin_mean[i])

        left_bin_std[i] = interval_std[0] + i * len_interval_std / n_bins
        rigtht_bin_std[i] = interval_std[0] + (i + 1) * len_interval_std / n_bins
        center_bin_std[i] = 0.5 * (left_bin_std[i] + rigtht_bin_std[i])

        left_bin_amp[i] = interval_amp[0] + i * len_interval_amp / n_bins
        rigtht_bin_amp[i] = interval_amp[0] + (i + 1) * len_interval_amp / n_bins
        center_bin_amp[i] = 0.5 * (left_bin_amp[i] + rigtht_bin_amp[i])

    if est_num == 1:
        single_gaussian = []
        weigth_single_gaussian = []
        for _p in particle:
            if _p.n_gaus == est_num:
                single_gaussian = np.append(single_gaussian, _p.gaussian[0])
                weigth_single_gaussian = np.append(weigth_single_gaussian, _p.weight)

        expected_mean = 0
        expected_std = 0
        expected_amp = 0

        weigth_single_gaussian /= np.sum(weigth_single_gaussian)

        for idx in range(len(single_gaussian)):
            expected_mean += single_gaussian[idx].mean * weigth_single_gaussian[idx]
            expected_std += single_gaussian[idx].std * weigth_single_gaussian[idx]
            expected_amp += single_gaussian[idx].amp * weigth_single_gaussian[idx]

        weight_bin_mean = np.zeros(n_bins)
        weight_bin_std = np.zeros(n_bins)
        weight_bin_amp = np.zeros(n_bins)

        for i in range(n_bins):
            for idx in range(len(single_gaussian)):
                if left_bin_mean[i] <= single_gaussian[idx].mean <= rigtht_bin_mean[i]:
                    weight_bin_mean[i] += weigth_single_gaussian[idx]

                if left_bin_std[i] <= single_gaussian[idx].std <= rigtht_bin_std[i]:
                    weight_bin_std[i] += weigth_single_gaussian[idx]

                if left_bin_amp[i] <= single_gaussian[idx].amp <= rigtht_bin_amp[i]:
                    weight_bin_amp[i] += weigth_single_gaussian[idx]

        map_mean = center_bin_mean[np.argmax(weight_bin_mean)]
        map_std = center_bin_std[np.argmax(weight_bin_std)]
        map_amp = center_bin_amp[np.argmax(weight_bin_amp)]

        if expected_mean <= 0:
            expected_error_mean = np.abs(expected_mean - (-2))
            expected_error_std = np.abs(expected_std - 1)
            expected_error_amp = np.abs(expected_amp - 0.7)
        else:
            expected_error_mean = np.abs(expected_mean - 2)
            expected_error_std = np.abs(expected_std - 0.5)
            expected_error_amp = np.abs(expected_amp - 0.3)

        if map_mean <= 0:
            error_mean = np.abs(map_mean - (-2))
            error_std = np.abs(map_std - 1)
            error_amp = np.abs(map_amp - 0.7)

        else:
            error_mean = np.abs(map_mean - 2)
            error_std = np.abs(map_std - 0.5)
            error_amp = np.abs(map_amp - 0.3)

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

        expected_mean_1 = 0
        expected_mean_2 = 0

        expected_std_1 = 0
        expected_std_2 = 0

        expected_amp_1 = 0
        expected_amp_2 = 0

        for idx in range(len(first_gaussian)):
            expected_mean_1 += first_gaussian[idx].mean * weigth_two_gaussian_1[idx]
            expected_std_1 += first_gaussian[idx].std * weigth_two_gaussian_1[idx]
            expected_amp_1 += first_gaussian[idx].amp * weigth_two_gaussian_1[idx]

        for idx in range(len(second_gaussian)):
            expected_mean_2 += second_gaussian[idx].mean * weigth_two_gaussian_2[idx]
            expected_std_2 += second_gaussian[idx].std * weigth_two_gaussian_2[idx]
            expected_amp_2 += second_gaussian[idx].amp * weigth_two_gaussian_2[idx]

        expected_mean = np.array([expected_mean_1, expected_mean_2])
        expected_std = np.array([expected_std_1, expected_std_2])
        expected_amp = np.array([expected_amp_1, expected_amp_2])

        weight_bin_mean_1 = np.zeros(n_bins)
        weight_bin_mean_2 = np.zeros(n_bins)

        weight_bin_std_1 = np.zeros(n_bins)
        weight_bin_std_2 = np.zeros(n_bins)

        weight_bin_amp_1 = np.zeros(n_bins)
        weight_bin_amp_2 = np.zeros(n_bins)

        for i in range(n_bins):
            for idx in range(len(first_gaussian)):
                if left_bin_mean[i] <= first_gaussian[idx].mean <= rigtht_bin_mean[i]:
                    weight_bin_mean_1[i] += weigth_two_gaussian_1[idx]

                if left_bin_std[i] <= first_gaussian[idx].std <= rigtht_bin_std[i]:
                    weight_bin_std_1[i] += weigth_two_gaussian_1[idx]

                if left_bin_amp[i] <= first_gaussian[idx].amp <= rigtht_bin_amp[i]:
                    weight_bin_amp_1[i] += weigth_two_gaussian_1[idx]

            for idx in range(len(second_gaussian)):
                if left_bin_mean[i] <= second_gaussian[idx].mean <= rigtht_bin_mean[i]:
                    weight_bin_mean_2[i] += weigth_two_gaussian_2[idx]

                if left_bin_std[i] <= second_gaussian[idx].std <= rigtht_bin_std[i]:
                    weight_bin_std_2[i] += weigth_two_gaussian_2[idx]

                if left_bin_amp[i] <= second_gaussian[idx].amp <= rigtht_bin_amp[i]:
                    weight_bin_amp_2[i] += weigth_two_gaussian_2[idx]

        map_mean = np.array(
            [center_bin_mean[np.argmax(weight_bin_mean_1)], center_bin_mean[np.argmax(weight_bin_mean_2)]])
        map_std = np.array([center_bin_std[np.argmax(weight_bin_std_1)], center_bin_std[np.argmax(weight_bin_std_2)]])
        map_amp = np.array([center_bin_amp[np.argmax(weight_bin_amp_1)], center_bin_amp[np.argmax(weight_bin_amp_2)]])

        expected_error_mean = np.sum(np.abs(expected_mean - np.array([-2, 2])))
        error_mean = np.sum(np.abs(map_mean - np.array([-2, 2])))

        expected_error_std = np.sum(np.abs(expected_std - np.array([1, 0.5])))
        error_std = np.sum(np.abs(map_std - np.array([1, 0.5])))

        expected_error_amp = np.sum(np.abs(expected_amp - np.array([0.7, 0.3])))
        error_amp = np.sum(np.abs(map_amp - np.array([0.7, 0.3])))

    return expected_error_mean, error_mean, expected_error_std, error_std, expected_error_amp, error_amp


def error_noise_std(particle, interval, n_bins, true_noise):
    len_interval = np.abs(interval[-1] - interval[0])
    left_bin = np.zeros(n_bins)
    rigtht_bin = np.zeros(n_bins)
    center_bin = np.zeros(n_bins)
    weight_bin = np.zeros(n_bins)
    cm_theta = 0
    for i in range(n_bins):
        left_bin[i] = interval[0] + i * len_interval / n_bins
        rigtht_bin[i] = interval[0] + (i + 1) * len_interval / n_bins
        center_bin[i] = 0.5 * (left_bin[i] + rigtht_bin[i])
        for _p in particle:
            cm_theta += _p.weight * _p.noise_std
            if left_bin[i] <= _p.noise_std <= rigtht_bin[i]:
                weight_bin[i] += _p.weight

    map_theta = center_bin[np.argmax(weight_bin)]

    return np.abs(map_theta - true_noise), np.abs(cm_theta - true_noise)
