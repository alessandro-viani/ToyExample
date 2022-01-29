import pickle

import numpy as np

from Util import eval_kl_div, error_parameters, error_noise_std

n_file = 1
kl_div = np.zeros(n_file)
folder_name = 'performance_analysis/'

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

interval_amp = [0.1, 10]
interval_mean = [-5, 5]
interval_std = [0.1, 10]
n_bins = 300

for idx in range(n_file):
    with open(f'{folder_name}data_{idx}.pkl', 'rb') as f:
        sourcespace, data, _n = pickle.load(f)
    with open(f'{folder_name}posterior_proposed_{idx}.pkl', 'rb') as f:
        post_prop = pickle.load(f)
    with open(f'{folder_name}posterior_classical_{idx}.pkl', 'rb') as f:
        post_clas = pickle.load(f)

    est_n_prop[idx] = post_prop.est_num_avg
    cm_m_clas[idx], map_m_prop[idx], cm_s_prop[idx], \
    map_s_prop[idx], cm_a_prop[idx], map_a_prop[idx] = error_parameters(post_prop.particle_avg,
                                                                        interval_mean, interval_std,
                                                                        interval_amp, n_bins,
                                                                        post_prop.est_num_avg,
                                                                        name_file=f'_prop_{idx}')

    map_n_clas[idx] = np.abs(np.argmax(post_prop.noise_posterior) - _n)
    cm_n_clas[idx] = np.abs(np.sum(post_prop.noise_posterior * post_prop.all_noise_std) - _n)

    est_n_clas[idx] = post_clas.est_num
    cm_m_clas[idx], map_m_clas[idx], cm_s_clas[idx], \
    map_s_clas[idx], cm_a_clas[idx], map_a_clas[idx] = error_parameters(post_clas.particle,
                                                                        interval_mean, interval_std,
                                                                        interval_amp, n_bins,
                                                                        post_prop.est_num,
                                                                        name_file=f'_clas_{idx}')

    map_n_clas[idx], cm_n_clas[idx] = error_noise_std(post_clas.particle, interval=[np.min(post_clas.vector_noise_std),
                                                                                    np.max(post_clas.vector_noise_std)],
                                                      n_bins=100, true_noise=_n)

    kl_div[idx] = eval_kl_div(post_clas.noise_posterior, post_prop.noise_posterior)

with open(f'{folder_name}analitics.pkl', 'wb') as f:
    pickle.dump([kl_div, est_n_clas, est_n_prop,
                 map_m_clas, map_s_clas, map_a_clas, map_n_clas,
                 map_m_prop, map_s_prop, map_a_prop, map_n_prop,
                 cm_m_clas, cm_s_clas, cm_a_clas, cm_n_clas,
                 cm_m_prop, cm_s_prop, cm_a_prop, cm_n_prop], f)
