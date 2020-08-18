from random import randrange, seed, randint

import matplotlib.pyplot as plt
from numpy import arange, array, std, mean
from scipy.special import binom

from quick_tests.haar_random_test_data import *


m = 8
n = 4
l = 2

outcomes_number = binom(m + l - 1, l)

evolution_means = means_fixed_5k_m8_n4_l2
evolution_stdevs = stdevs_fixed_5k_m8_n4_l2
expected_probabilities = 1.0 / outcomes_number

print(expected_probabilities)
matrices_number = len(evolution_means)

plt.loglog(range(matrices_number), evolution_means)
plt.loglog(range(len(evolution_stdevs)), evolution_stdevs, color='green')
plt.loglog(range(len(evolution_means)), [expected_probabilities for _ in range(len(evolution_means))], color='red')
plt.show()

'''
x = []
y = []
y_std = []
vals = []
vals_std = []

j = randint(0, 1)
k = randint(0, 1)

for i in range(20000):
    x.append(i)
    m = generate_haar_random_unitary_matrix_mezzardi(2)
    vals.append(m[j][k])
    y.append(abs(mean(vals)))
    y_std.append(std(y))
    vals_std.append(std(vals))

plt.plot(x, y)
plt.plot(x, y_std, color='red')
plt.plot(x, vals_std, color='green')
plt.show()
'''
