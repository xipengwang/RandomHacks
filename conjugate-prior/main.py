import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import time

success_prob = 0.3
data = np.random.binomial(n=1, p=success_prob, size=4000)

theta_range = np.linspace(0, 1, 1000)
a = 20
b = 80

tic = time.time()
theta_range_e = theta_range + 0.0001
prior = stats.beta.cdf(x = theta_range_e, a=a, b=b) - stats.beta.cdf(x = theta_range, a=a, b=b)
likelihood = stats.binom.pmf(k = np.sum(data), n = len(data), p = theta_range)
posterior = likelihood * prior
normalized_posterior = posterior / np.sum(posterior)
toc = time.time()
ms_naive = 1000.0 * (toc - tic)
print(f'MAP from numerical optimization: {theta_range[np.argmax(normalized_posterior)]}, time cost:{ms_naive}')

# Use conjugate-prior
tic = time.time()
a = a+np.sum(data)
b = b+len(data)-np.sum(data)
posterior = stats.beta.pdf(x = theta_range, a=a, b=b)
mode = (a-1) / (a+b-2)
toc = time.time()
ms_naive = 1000.0 * (toc - tic)
print(f'MAP from using beta distribution as conjugate prior: {mode}, time cost:{ms_naive}')

# Plotting the prior distribution
plt.rcParams['figure.figsize'] = [20, 7]
fig, ax = plt.subplots()
plt.plot(theta_range, posterior, linewidth=3)
plt.title('posterior density (beta distribution)')
plt.xlabel('theta', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.show()
