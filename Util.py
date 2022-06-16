# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:48:19 2022

@author: viani
"""
import numpy as np
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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

def eval_kl_div(post_clas, post_prop):
    post_clas = post_clas / np.sum(post_clas)
    post_prop = post_prop / np.sum(post_prop)
    return np.sum(post_clas * np.log(np.divide(post_clas, post_prop)))

def bin_creation(interval, n_bins):
    left_bin = np.zeros(n_bins)
    right_bin = np.zeros(n_bins)
    center_bin = np.zeros(n_bins)

    for i in range(n_bins):
        left_bin[i] = interval[0] + i * np.abs(interval[-1] - interval[0]) / n_bins
        right_bin[i] = interval[0] + (i + 1) * np.abs(interval[-1] - interval[0]) / n_bins
        center_bin[i] = 0.5 * (left_bin[i] + right_bin[i])

    return left_bin, center_bin, right_bin


def analytics(n_file=1, interval_mean=None, n_bins=None, folder_name='solution/'):

    kl_div = np.zeros(n_file)

    cpu_time_clas = np.zeros(n_file)
    cpu_time_prop = np.zeros(n_file)

    var_clas = np.zeros(n_file)
    var_prop = np.zeros(n_file)

    err_map_mean_clas = np.zeros(n_file)
    err_map_theta_clas = np.zeros(n_file)

    err_map_mean_prop = np.zeros(n_file)
    err_map_theta_prop = np.zeros(n_file)

    err_cm_mean_clas = np.zeros(n_file)
    err_cm_theta_clas = np.zeros(n_file)

    err_cm_mean_prop = np.zeros(n_file)
    err_cm_theta_prop = np.zeros(n_file)

    for idx in range(n_file):
        print(f'iter:{idx+1}/{n_file}')
        with open(f'data/data_{idx}.pkl', 'rb') as f:
            sourcespace, data, _n = pickle.load(f)
        with open(f'{folder_name}sol_prop_{idx}.pkl', 'rb') as f:
            post_prop = pickle.load(f)
        with open(f'{folder_name}sol_clas_{idx}.pkl', 'rb') as f:
            post_clas = pickle.load(f)

        cpu_time_clas[idx] = post_clas.cpu_time
        cpu_time_prop[idx] = post_prop.cpu_time

        var_prop[idx] = np.sum(post_prop.noise_posterior * np.square(post_prop.all_noise_std -
                                                     np.sum(post_prop.noise_posterior * post_prop.all_noise_std)))
        var_clas[idx] = np.sum(post_clas.vector_weight * np.square(post_clas.vector_noise_std -
                                                     np.sum(np.asarray(post_clas.vector_noise_std) * np.asarray(post_clas.vector_weight))))

        interval_theta = [np.min(post_prop.all_noise_std), np.max(post_prop.all_noise_std)]

        err_cm_mean_prop[idx], err_map_mean_prop[idx], err_cm_theta_prop[idx], err_map_theta_prop[idx] =\
            err_parameters(post_prop, interval_mean, interval_theta, n_bins, true_noise=_n, prop=True)

        err_cm_mean_clas[idx], err_map_mean_clas[idx], err_cm_theta_clas[idx], err_map_theta_clas[idx] =\
            err_parameters(post_clas, interval_mean, interval_theta, n_bins, true_noise=_n, prop=False)

        kl_div[idx] = eval_kl_div(post_clas.noise_posterior, post_prop.noise_posterior)

    with open(f'{folder_name}analitics.pkl', 'wb') as f:
        pickle.dump([cpu_time_clas, cpu_time_prop,
                     kl_div,
                     var_clas, var_prop,
                     err_map_mean_clas, err_map_mean_prop,
                     err_map_theta_clas, err_map_theta_prop,
                     err_cm_mean_clas, err_cm_mean_prop,
                     err_cm_theta_clas, err_cm_theta_prop], f)



def kl_plot(folder_name, saturation=0.7, linewidth = 1, width=0.7, dpi=100, plot_show=True):

    with open(f'{folder_name}analitics.pkl', 'rb') as f:
        cpu_time_clas, cpu_time_prop, \
        kl_div, \
        var_clas, var_prop, \
        err_map_mean_clas, err_map_mean_prop,\
        err_map_theta_clas, err_map_theta_prop, \
        err_cm_mean_clas, err_cm_mean_prop, \
        err_cm_theta_clas, err_cm_theta_prop = pickle.load(f)

    plt.figure(figsize=(6, 7), dpi=100)
    sns.boxplot(data=[np.log10(kl_div)], palette=['#1f77b4'], saturation=saturation, width=width, linewidth=linewidth)
    plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.legend([], frameon=False)
    plt.title(r'$\log_{10}(D_{KL})$')
    plt.tight_layout()
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tight_layout()

    plt.savefig('fig/kl_div.png', dpi=dpi)
    if plot_show:
        plt.show()
    else:
        plt.close()

    return 0

def var_plot(folder_name, saturation=0.7, linewidth=1, width=0.7, dpi=100, plot_show=True):
    with open(f'{folder_name}analitics.pkl', 'rb') as f:
        cpu_time_clas, cpu_time_prop, \
        kl_div, \
        var_clas, var_prop, \
        err_map_mean_clas, err_map_mean_prop,\
        err_map_theta_clas, err_map_theta_prop, \
        err_cm_mean_clas, err_cm_mean_prop, \
        err_cm_theta_clas, err_cm_theta_prop = pickle.load(f)

    plt.figure(figsize=(6, 7), dpi=100)
    sns.boxplot(data=[var_clas, var_prop], palette=['red','#1f77b4'], saturation=saturation, width=width, linewidth=linewidth)
    plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.legend([], frameon=False)
    plt.title('Variance')
    plt.tight_layout()
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tight_layout()

    plt.savefig('fig/var_post.png', dpi=dpi)
    if plot_show:
        plt.show()
    else:
        plt.close()
    return 0

def theta_plot(folder_name, saturation=0.7, linewidth=1, width=0.7, dpi=100, plot_show=True):
    with open(f'{folder_name}analitics.pkl', 'rb') as f:
        cpu_time_clas, cpu_time_prop, \
        kl_div, \
        var_clas, var_prop, \
        err_map_mean_clas, err_map_mean_prop,\
        err_map_theta_clas, err_map_theta_prop, \
        err_cm_mean_clas, err_cm_mean_prop, \
        err_cm_theta_clas, err_cm_theta_prop = pickle.load(f)


    fig, ax = plt.subplots(1,2, figsize=(16, 9), dpi=100)
    sns.boxplot(data=[err_map_theta_clas, err_map_theta_prop], palette=['red','#1f77b4'], saturation=saturation, width=width, linewidth=linewidth, ax=ax[0])
    sns.boxplot(data=[err_cm_theta_clas, err_cm_theta_prop], palette=['red','#1f77b4'], saturation=saturation, width=width, linewidth=linewidth, ax=ax[1])

    plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.legend([], frameon=False)
    plt.xticks((0, 1), ('MAP', 'CM'))
    plt.title(r'error $\theta$')
    plt.tight_layout()

    plt.savefig('fig/theta_error.png', dpi=dpi)
    if plot_show:
        plt.show()
    else:
        plt.close()

    return 0

def error_parameter_plot(folder_name, size_ticks=22, fontsize=22, saturation=0.7, linewidth=1, width=0.7, dpi=100, plot_show=True):
    with open(f'{folder_name}analitics.pkl', 'rb') as f:
        cpu_time_clas, cpu_time_prop, \
        kl_div, \
        var_clas, var_prop, \
        err_map_mean_clas, err_map_mean_prop,\
        err_map_theta_clas, err_map_theta_prop, \
        err_cm_mean_clas, err_cm_mean_prop, \
        err_cm_theta_clas, err_cm_theta_prop = pickle.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=100)
    sns.boxplot(data=[err_map_mean_clas, err_map_mean_prop], palette=['red', '#1f77b4'],
                saturation=saturation, width=width,
                linewidth=linewidth,
                ax=ax[0])

    sns.boxplot(data=[err_cm_mean_clas, err_cm_mean_prop], palette=['red', '#1f77b4'],
                saturation=saturation, width=width,
                linewidth=linewidth,
                ax=ax[1])

    plt.sca(ax[0])
    plt.xticks(size=size_ticks)
    plt.yticks(size=size_ticks)
    ax[0].set(xlabel=None)
    ax[0].set(ylabel=None)
    ax[0].set(xticklabels=[])
    ax[0].tick_params(bottom=False)
    plt.legend([], [], frameon=False)
    plt.title(r'MAP error mean', fontsize=fontsize)
    plt.sca(ax[1])
    plt.xticks(size=size_ticks)
    plt.yticks(size=size_ticks)
    ax[1].set(xlabel=None)
    ax[1].set(ylabel=None)
    ax[1].set(xticklabels=[])
    ax[1].tick_params(bottom=False)
    plt.legend([], [], frameon=False)
    plt.title(r'CM error mean', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('fig/error_parameters_toy.png', dpi=dpi)
    if plot_show:
        plt.show()
    else:
        plt.close()
    return 0

def cpu_time_plot(folder_name, size_ticks=22, fontsize=22, saturation=0.7, linewidth=1, width=0.7, dpi=100, plot_show=True):
    with open(f'{folder_name}analitics.pkl', 'rb') as f:
        cpu_time_clas, cpu_time_prop, \
        kl_div, \
        var_clas, var_prop, \
        err_map_mean_clas, err_map_mean_prop,\
        err_map_theta_clas, err_map_theta_prop, \
        err_cm_mean_clas, err_cm_mean_prop, \
        err_cm_theta_clas, err_cm_theta_prop = pickle.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=100)
    sns.boxplot(data=[cpu_time_clas, cpu_time_prop], palette=['red', '#1f77b4'],
                saturation=saturation, width=width,
                linewidth=linewidth)
    plt.savefig('fig/cpu_time.png', dpi=dpi)
    if plot_show:
        plt.show()
    else:
        plt.close()

    return 0

def all_plots(folder_name, size_ticks=22, fontsize=22, saturation=0.7, linewidth=1, width=0.7, dpi=100, plot_show=True):
    kl_plot(folder_name, saturation=saturation, linewidth=linewidth, width=width, dpi=dpi, plot_show=plot_show)
    theta_plot(folder_name, saturation=saturation, linewidth=linewidth, width=width, dpi=dpi, plot_show=plot_show)
    var_plot(folder_name, saturation=saturation, linewidth=linewidth, width=width, dpi=dpi, plot_show=plot_show)
    error_parameter_plot(folder_name, size_ticks=size_ticks, fontsize=fontsize, saturation=saturation, linewidth=linewidth,
                         width=width, dpi=dpi, plot_show=plot_show)
    cpu_time_plot(folder_name, size_ticks=size_ticks, fontsize=fontsize, saturation=saturation, linewidth=linewidth, width=width,
                  dpi=dpi, plot_show=plot_show)

    return 0




   # def theta_estimates(self, n_bins=50):
   #     integral1 = 0
   #     self.pm_smc = 0
   #     for idx, _n in enumerate(self.all_noise_std):
   #         integral1 += 0.5 * (self.noise_posterior[idx-1] + self.noise_posterior[idx]) * np.abs(self.all_noise_std[idx-1] - _n)

   #     for idx, _n in enumerate(self.all_noise_std):
   #         self.pm_smc += 0.5 * (self.all_noise_std[idx-1] * self.noise_posterior[idx-1] +
   #                                           _n * self.noise_posterior[idx]) * np.abs(self.all_noise_std[idx-1] - _n)

   #     self.pm_smc /= integral1

   #     self.map_smc = self.all_noise_std[np.argmax(self.noise_posterior)]
   #     self.ml_smc = self.all_noise_std[np.argmax(self.noise_likelihood)]
   #     self.chosen_iter = np.argmax(self.noise_likelihood)+2

   #     #spline estimates
   #     self.grid_theta = np.unique(
   #                         np.sort(
   #                             np.append(self.all_noise_std,
   #                                       np.linspace(np.min(self.all_noise_std), np.max(self.all_noise_std), int(self.point_spline)))))

   #     self.noise_spline_posterior = scipy.interpolate.splev(self.grid_theta,
   #                                                           scipy.interpolate.splrep(self.all_noise_std[::-1], self.noise_posterior[::-1]))
   #     self.noise_spline_likelihood = scipy.interpolate.splev(self.grid_theta,
   #                                                             scipy.interpolate.splrep(self.all_noise_std[::-1], self.noise_likelihood[::-1]))

   #     self.map_spline = self.grid_theta[np.argmax(self.noise_spline_posterior)]
   #     self.ml_spline = self.grid_theta[np.argmax(self.noise_spline_likelihood)]

   #     integral = 0
   #     self.pm_spline = 0
   #     for idx, _n in enumerate(self.grid_theta):
   #         integral += 0.5*(self.noise_spline_posterior[idx-1] + self.noise_spline_posterior[idx]) * np.abs(self.grid_theta[idx-1] - _n)

   #     for idx, _n in enumerate(self.grid_theta):
   #         self.pm_spline += 0.5 * (self.grid_theta[idx-1] * self.noise_spline_posterior[idx-1] +
   #                                              _n * self.noise_spline_posterior[idx]) * np.abs(self.grid_theta[idx-1] - _n)

   #     self.pm_spline /= integral



   #     if not self.prop_method:
   #         left_bin_theta, center_bin_theta, right_bin_theta = bin_creation(np.array([np.min(self.vector_noise_std),
   #                                                                                    np.max(self.vector_noise_std)]),
   #                                                                          n_bins)
   #         weight_bin_theta = np.zeros(n_bins)
   #         self.pm_smc = np.sum(self.vector_noise_std * self.vector_weight)

   #         for idx in range(n_bins):
   #             for jdx, _n in enumerate(self.vector_noise):
   #                 if left_bin_theta[idx] <= _n <= right_bin_theta[idx]:
   #                     weight_bin_theta[idx] += self.vector_weight[jdx]
   #         self.map_smc = center_bin_theta[np.argmax(weight_bin_theta)]

   #     return 0
