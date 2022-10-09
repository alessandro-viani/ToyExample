import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


def log_normal(x, mean, std):
    return -np.log(np.sqrt(2 * np.pi) * std) - 0.5 * np.square((x - mean) / std)


def sequence_of_exponents(n_iter, max_exp):
    return np.concatenate((np.power(np.linspace(0, 1, n_iter), 4), [max_exp + 0.1]))


def creation_data(n_data, theta):
    sourcespace = np.linspace(-5, 5, n_data)
    data = np.zeros(int(n_data))
    for i in range(0, n_data):
        data[i] = 1 * stats.norm.pdf(sourcespace[i], 0, 1) + np.random.normal(0, theta)
    return sourcespace, data


def eval_kl_div(post_clas, post_prop):
    post_clas = post_clas / np.sum(post_clas)
    post_prop = post_prop / np.sum(post_prop)
    return np.sum(post_clas * np.log(np.divide(post_clas, post_prop)))


def bin_creation(min_interval, max_interval, n_bins):
    left_bin = np.zeros(n_bins)
    right_bin = np.zeros(n_bins)
    center_bin = np.zeros(n_bins)

    for i in range(n_bins):
        left_bin[i] = min_interval + i * np.abs(max_interval - min_interval) / n_bins
        right_bin[i] = min_interval + (i + 1) * np.abs(max_interval - min_interval) / n_bins
        center_bin[i] = 0.5 * (left_bin[i] + right_bin[i])

    return left_bin, center_bin, right_bin


def plot_confront(post_pm, post_fb, post_em):
    alpha = 0.5
    sns.set_style('darkgrid')
    color_map = ['#1f77b4', 'darkorange', 'forestgreen', 'red']

    min_mu = np.min([np.min(post_pm.vector_mean), np.min(post_fb.vector_mean), np.min(post_em.vector_mean)])
    min_theta = np.min([np.min(post_pm.vector_theta), np.min(post_fb.vector_theta), np.min(post_em.vector_theta)])
    max_mu = np.max([np.max(post_pm.vector_mean), np.max(post_fb.vector_mean), np.max(post_em.vector_mean)])
    max_theta = np.max([np.max(post_pm.vector_theta), np.max(post_fb.vector_theta), np.max(post_em.vector_theta)])

    post_fb.grid_theta = post_pm.grid_theta
    post_fb.theta_posterior = stats.gaussian_kde(post_fb.vector_theta, weights=post_fb.vector_weight).pdf(
        post_fb.grid_theta)
    integral = 0.5 * np.sum((post_fb.theta_posterior[:-1] + post_fb.theta_posterior[1:]) * np.abs(
        post_fb.grid_theta[:-1] - post_fb.grid_theta[1:]))
    post_fb.theta_posterior /= integral

    fig, ax = plt.subplots(3, 2, figsize=(16, 9))

    plt.sca(ax[0, 0])
    plt.xlim([min_mu, max_mu])
    plt.title(r'$p(\mu\mid y)$')
    plt.ylabel('Proposed Method', fontsize=10, rotation=90, labelpad=20)
    sns.histplot(x=post_pm.vector_mean, stat='probability', weights=post_pm.vector_weight, bins=post_pm.n_bins,
                 color=color_map[0], alpha=alpha)

    plt.sca(ax[0, 1])
    plt.xlim([min_theta, max_theta])
    plt.title(r'$p(\theta\mid y)$')
    plt.plot(post_pm.grid_theta, post_pm.theta_posterior, color=color_map[0], alpha=alpha)
    plt.fill_between(post_pm.grid_theta, post_pm.theta_posterior, color=color_map[0], alpha=alpha * 0.25)

    plt.sca(ax[1, 0])
    plt.xlim([min_mu, max_mu])
    plt.ylabel('Fully Bayesian', fontsize=10, rotation=90, labelpad=20)
    sns.histplot(x=post_fb.vector_mean, stat='probability', weights=post_fb.vector_weight, bins=post_fb.n_bins,
                 color=color_map[1], alpha=alpha)

    plt.sca(ax[1, 1])
    plt.xlim([min_theta, max_theta])
    plt.plot(post_fb.grid_theta, post_fb.theta_posterior, color=color_map[1], alpha=alpha)
    plt.fill_between(post_fb.grid_theta, post_fb.theta_posterior, color=color_map[1], alpha=alpha * 0.25)

    plt.sca(ax[2, 0])
    plt.xlim([min_mu, max_mu])
    plt.ylabel('Expectation Maximization', fontsize=10, rotation=90, labelpad=20)
    sns.histplot(x=post_em.vector_mean, stat='probability', weights=post_em.vector_weight, bins=post_em.n_bins,
                 color=color_map[2], alpha=alpha)

    plt.sca(ax[2, 1])
    plt.xlim([min_theta, max_theta])
    plt.plot(post_em.grid_theta, post_em.theta_posterior, color=color_map[2], alpha=alpha)
    plt.fill_between(post_em.grid_theta, post_em.theta_posterior, color=color_map[2], alpha=alpha * 0.25)
    fig.tight_layout()
