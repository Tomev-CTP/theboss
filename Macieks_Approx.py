# python3 script calculating problem of boson sampling with loss of
# particles in exact and approximate way as suggested in Clifford &
# Clifford 2017 and Oszmaniec & Brod 2018
# author of the code: Maciej Kościelski - koscielski.maciej@gmail.com

import numpy as np
import random as rnd
from scipy import special
import itertools
import math
import matplotlib.pyplot as plt
import sys

from src.Boson_Sampling_Utilities import calculate_permanent, particle_state_to_modes_state,\
    modes_state_to_particle_state
from src.Quantum_Computations_Utilities import generate_haar_random_unitary_matrix, \
    calculate_total_variation_distance, calculate_distance_between_matrices

from src.BosonSamplingSimulator import BosonSamplingSimulator


# =====================================================================

# the matrix to be ''permanented''
def U_Sr_matrix(U, S, r, m, n):
    U_S = 1j * np.zeros((m, n))
    column = 0

    # copiyng s_i times the i-th column of U
    for i in range(m):

        s_i = S[i]
        while s_i > 0:
            U_S[:, column] = U[:, i]
            column += 1
            s_i -= 1

    U_Sr = 1j * np.zeros((n, n))
    row = 0

    # copiyng r_i times the i-th row of U_S
    for i in range(m):

        r_i = r[i]
        while r_i > 0:
            U_Sr[row, :] = U_S[i, :]
            row += 1
            r_i -= 1

    return U_Sr


# all the possible states from 'S' with 'No' photons
def generate_lossy_inputs(n, S, No):
    x0 = []
    for i in range(m):
        for j in range(int(S[i])):
            x0.append(i)  # modes numbered from 0

    table = []

    # symmetrization
    for obj in itertools.combinations(list(range(n)), No):
        x = []  # all possible entries with No of photons from S
        for el in obj:
            x.append(x0[el])

        S_No = particle_state_to_modes_state(x, m)
        if (all(list(table[el]) != list(S_No) for el in range(len(table)))):
            table.append(S_No)

    return table


# probability of 'r' for indistinguishable photons
def prob(n, U, S, r, No):
    # numbers of modes from zero

    r_new = list(r[:])  # to create a new list
    T_No = particle_state_to_modes_state(r_new, m)  # proposed sample in modes basis

    prob = 0.
    # symmetrization of the input
    S_No_table = generate_lossy_inputs(n, S, No)
    for S_No in S_No_table:
        # probability in modes-basis (read Brod & Oszmaniec 2019)
        subprob = abs(calculate_permanent(U_Sr_matrix(U, S_No, T_No, m, No))) ** 2
        for S_j in S_No:
            subprob /= math.factorial(S_j)

        prob += subprob

    # normalization (read Brod & Oszmaniec 2019)
    prob /= math.factorial(No)
    prob /= special.binom(n, No)

    return prob


# randomizing the input state
def random_phases(state):
    # the input in modes-basis
    state_rand = []

    # random phase for every mode
    for mode in state:
        phi = rnd.random() * 2 * np.pi
        state_rand.append(np.exp(1j * phi) * mode)

    return state_rand


# approximate simulation of boson sampling according to Brod & Oszmaniec 2018
def approximate_simulation(input_state, photons_left):
    # from the 'input state' (in modes-basis) returns state with 'photons_left' photons and probability of having this
    # state (for fixed loss ratio) according to distribution given by 'U' matrix, with some error
    # input_state - one of basis vectors

    # --- the input state --------------------------------------------
    psi0 = input_state / np.sqrt(n)
    psi0 = random_phases(psi0)  # randomization

    # --- measuring the resulting state ------------------------------
    # a simple evolution by the U operator
    psi_out = np.dot(U, psi0)

    probabilities = []
    output = np.zeros(m)
    for detector in psi_out:
        probabilities.append(np.conjugate(detector) * detector)
    # print( np.sum(probabilities) )

    # evolution of every photon is independent of the others
    l = photons_left
    for photon in range(l):
        x = rnd.random()
        i = 0
        prob = probabilities[i]
        while (x > prob):
            i += 1
            prob += probabilities[i]
        output[i] += 1

    return output


# probability of obtaining the sample 'T' from 'S' in exact simulation
def probability_of_sample(n, S, T, l):
    # the results are to be checked

    # computing the probability of given output
    r = modes_state_to_particle_state(T, l)

    p = prob(n, U, S, r, l)

    # different states in particles-basis may give the same state T
    p *= math.factorial(l)
    for i in range(m):
        p /= math.factorial(T[i])

    return p


# generating 'reps' samples and exact probabilities of the results
def repeat_experiments(S, l, reps, binning=False):
    results = []

    # running 'experiment' many times
    for i in range(reps):
        # r, p_k = exact_simulation( S, l )
        # generation of a saple using the closest separable state
        T_approx = approximate_simulation(S, l)

        results.append(T_approx)

    # counting the resulted outputs to calculate prob. distribution
    p_max, p_exact_max, p_exact_cumulated = 0., 0., 0.
    table_of_results, table_of_approx_ps, table_of_ps = [], [], []
    for result in results:

        # loop over all new results
        if (all(list(result) != list(element) for element in table_of_results)):

            # counting the results
            p = 0.
            for another_result in results:
                if (list(result) == list(another_result)):
                    p += 1. / reps

            p_exact = probability_of_sample(n, S, result, l)  # exact probability

            # the most probable result
            if (p_max < p):
                p_max = p
                p_exact_max = p_exact

            # saving data
            table_of_results.append(list(result))
            table_of_approx_ps.append(p)
            table_of_ps.append(probability_of_sample(n, S, result, l))
            p_exact_cumulated += p_exact

    if (p_exact_cumulated < 0.95):
        print("it won't be a good sample!", "rise the number of repetitions!")
        pass

    if binning:
        return [p_max, 1 - p_max], [p_exact_max, 1 - p_exact_max], p_exact_cumulated
    else:
        return table_of_approx_ps, table_of_ps, p_exact_cumulated


# TVD for incomplete sets of samples
def compare(table_of_approx_ps, table_of_ps, p_exact_cumulated):
    # calculating the TVD
    distance = calculate_total_variation_distance(table_of_approx_ps, table_of_ps)
    # adding the distance to the results which did not appear
    distance += (1. - p_exact_cumulated) / 2.

    return distance


# all the possible outputs with 'No' photons
def generate_outputs(l):
    outcomes = []

    outcome = np.zeros(m)
    outcome[0] = l
    outcomes.append(outcome)

    # a loop generating new possible outcomes
    while (outcomes[-1][m - 1] != l):

        k = m - 1
        while (outcomes[-1][k - 1] == 0):
            k -= 1

        outcome = outcomes[-1].copy()
        outcome[k - 1] -= 1
        outcome[k:] = 0
        outcome[k] = l - sum(outcome)

        outcomes.append(outcome)

    return outcomes


# all the inputs with 'l' particles on 'n' modes
def generate_n_mode_inputs(m, n, l):
    # n has to be lower than m !!!

    inputs = []

    n_input = np.zeros(n)
    n_input[0] = l
    m_input = np.zeros(m)
    m_input[0] = l
    inputs.append(m_input)

    # a loop generating new possible inputs
    while (n_input[n - 1] != l):

        k = n - 1
        while (n_input[k - 1] == 0):
            k -= 1

        n_input[k - 1] -= 1
        n_input[k:] = 0
        n_input[k] = l - sum(n_input)

        m_input = np.zeros(m)
        m_input[:n] = list(n_input[:])

        inputs.append(m_input)

    return inputs


# distribution of results for the 'twirled state' in the input
def generate_approximate_distribution():
    prob_table = []
    possible_outputs = generate_outputs(l)
    # input states in the basis of particles numbers
    possible_inputs = generate_n_mode_inputs(m, n, l)
    print(possible_inputs)

    # all the outputs in particle numbers
    for output in possible_outputs:

        # probability of given state at the output
        probability_of_state = 0.

        # every state with equal weight ???
        for input_state in possible_inputs:

            # print( 'input state: ', input_state )
            subprob = probability_of_sample(l, input_state, output, l)

            # the 'l over tau' factor
            subprob *= math.factorial(l)
            for S_i in input_state:
                subprob /= math.factorial(S_i)

            probability_of_state += subprob

        probability_of_state /= n ** l
        prob_table.append(probability_of_state)

    # returning probability distribution
    return prob_table


# distribution of results for the 'twirled state' in the input
def generate_exact_distribution(U):
    prob_table = []
    possible_outputs = generate_outputs(l)

    # all the outputs in particle numbers
    for output in possible_outputs:
        # probability of given state at the output
        probability_of_state = probability_of_sample(n, S, output, l)
        prob_table.append(probability_of_state)

    # returning probability distribution
    return prob_table


# =====================================================================

#######################################################################
# --- initial parameters ----------------------------------------------
n = 4  # number of photons
# m = n**2 # number of modes
m = 8  # number of modes
#######################################################################

whole_distr = False  # calculate the whole distribution
simulate = True  # sample the distribution
binning = False  # binn results to compare with exact distribution

# the size of the problem
print('number of possible outputs in order of',
      BosonSamplingSimulator.calculate_number_of_outcomes_with_l_particles_in_m_modes(m, n))

# reps = 500 # number of samples
excess = 10  # (number of samples)/(no. of possible results)

# lists for storing TVDs
plots = []

#######################################################################
repetitions = 5  # number of different U matrices
#######################################################################
# repeat for few different U matrices
for repeat in range(repetitions):

    l_plot, simulation_plot, TVD_plot = [], [], []

    # generation of the experimental setting
    U = generate_haar_random_unitary_matrix(m)  # some random unitary evolution matrix
    # U = np.identity( m ) # some random unitary evolution matrix
    dist_UI = calculate_distance_between_matrices(U, np.identity(m))
    print('||U - I|| =', dist_UI)

    # loop over all possible numbers of photons left
    for l in range(1, n + 1):

        print('m =', m, 'n =', n, 'l =', l)
        print('gives the bound',
              BosonSamplingSimulator
              .calculate_distance_from_lossy_bosonic_n_particle_state_to_set_of_symmetric_separable_l_particles_states(
                  n, l), 'for TVD')  # Theorem 1 from Oszmaniec & Brod 2018

        # checking the input parameters
        if n > m:
            sys.exit("n should be smaller than m")
        if l > n:
            sys.exit("there is not so many photons at the input")

        # --- the input state if(whole_distr): the state is fixed --------
        S = np.zeros(m)
        S[:n] = 1  # number of photons in each mode at input

        # --- generating results -----------------------------------------
        # comparing distributions in exact and approximate simulations
        if (whole_distr):

            # distribution from 'twirled' state
            approximate_distr = generate_approximate_distribution()

            # remove it:
            if (abs(sum(approximate_distr) - 1.) > 0.001):
                print('probabilities does not sum to one !!!')
                print('but to', sum(approximate_distr))

            # exact distribution for input (1,...,1,0,...,0)
            S = np.zeros(m)
            S[:n] = 1  # number of photons in each mode at input
            exact_distr = generate_exact_distribution(U)

            # remove it:
            if (abs(sum(exact_distr) - 1.) > 0.001):
                print('probabilities does not sum to one !!!')
                print('but to', sum(exact_distr))

            # TVD between exact distribution and the one
            distance = calculate_total_variation_distance(approximate_distr, exact_distr)

        # sampling using approximate simulation scheme
        elif (simulate):
            # this branch for binning==True may not work

            # number of samples
            reps = excess * BosonSamplingSimulator.calculate_number_of_outcomes_with_l_particles_in_m_modes(m, l)

            # when binning==True then only two bins used
            # table_of_approx_ps_0, table_of_ps, p_exact_cumulated_0 = repeat_experiments(S, l, reps, binning)
            table_of_approx_ps, table_of_ps, p_exact_cumulated = repeat_experiments(S, l, reps, binning)

            print(table_of_approx_ps)
            print(table_of_ps)
            print(p_exact_cumulated)

            # TVD between experimental distribution and the exact one
            distance = compare(table_of_approx_ps, table_of_ps, p_exact_cumulated)
            print('TVD =', distance, 'p_exact_cumulated =', int(p_exact_cumulated * 1000) / 1000.)

        # saving data to make a plot
        l_plot.append(l)
        simulation_plot.append(distance)
        TVD_plot.append(
            BosonSamplingSimulator.calculate_distance_from_lossy_bosonic_n_particle_state_to_set_of_symmetric_separable_l_particles_states(
                n, l))

    # here are all the results of the calculations
    plots.append([l_plot, simulation_plot, dist_UI])

# saving plot of TVDs in a file
plt.plot(l_plot, TVD_plot, color='orange')
# drawing al the results
ax = plt.subplot(111)  # I don't know why 111...
for plot in plots:
    lbl = '|UI|=' + str('%.2f' % plot[2])
    plt.scatter(plot[0], plot[1], label=lbl)
ax.legend()
ax.set_xlabel('l - number of photons in the output')
ax.set_ylabel('TVD between exact distr. and approximate one')
# plt.show()
# name = '|UI|='+str('%.2f'%dist_UI)+'_n='+str(int(n))+'_m='+str(int(m))+'.png'
name = 'n=' + str(int(n)) + '_m=' + str(int(m)) + '.png'
plt.savefig(name, format="png", dpi=200)
