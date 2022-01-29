import pickle

import matplotlib.pyplot as plt

with open('save_folder/posterior.pkl', 'rb') as f:
    post = pickle.load(f)

fig, ax = plt.subplots(1, 2)
ax[0].hist(post.vector_mean, weights=post.vector_weight, bins=50)
ax[1].plot(post.all_noise_std, post.noise_posterior)
plt.show()
