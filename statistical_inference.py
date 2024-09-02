import numpy as np
import math
import random

from numba import jit

# creates an initial state where q denotes the share of down spins. So q = 1, creates a configuration, where all spins
# point downwards

@jit(nopython=True)
def create_state(q,N):
    
    state = np.ones(N)
    
    for i in range(0,math.floor(q*N)):
        
        state[i]=-1.0
        
    np.random.shuffle(state)
    
    return state

@jit(nopython=True)
def create_random_field(N):
    
    """creates a random field h with mean 0 and variance 1/N

    Returns:
        field {array of floats}: field
    """    

    field = np.random.standard_normal(N)
    field = field/np.sqrt(N)

    return field

@jit(nopython=True)
def create_random_couplings(N):
    
    """creates random couplings J with mean 0 and variance 1/N

    Returns:
        _type_: _description_
    """
    
    couplings = np.zeros((N,N))

    for i in range(N):

        for j in range(N):

            if i > j:

                couplings[i,j] = couplings[j,i]
            
            else:

                couplings[i,j] = np.random.normal(loc = 0, scale = 1/np.sqrt(N)) 
    
    np.fill_diagonal(couplings,0)

    return couplings

def test_randomness_of_couplings():
    
    """tests if the variance and mean of the couplings and the field is correct

    Returns:
        mean: mean of the created couplings
        var: variance of the created couplings
    """
    
    couplings = []

    for i in range(100000):

        couplings.append(create_random_couplings(10))
    
    mean = np.mean(couplings, axis = 0)
    var = np.mean(np.power(couplings,2), axis = 0)

    return mean, var

#calculates the magnetisation

@jit(nopython=True)
def magnetisation(state):
    
    return np.sum(state)

#calculates the energy

@jit(nopython=True)
def energy(state,L,J,h):
    
    U = - np.sum(h*state)-0.5*np.sum(state*np.dot(J,state))
    
    return U

#calculate the energy change
# H = -h * s_i - J * s_j * s_i
# dH = H'-H = - h * (s'_i-s_i)-J*s_j*(s'_i-s_i)

@jit(nopython=True)
def energy_change(state,i,J,h):
    
    # dflip is (s'_i-s_i), which is either -2 if s_i was 1 and 2 if s_i was -1
    value = state[i]

    dspin = (-2)*value
    
    # energy change, that results from a spin flip
    
    # we need only to sum up the spins of the neighbours and multiply with dflip
    
    dH = -h[i]*dspin - np.sum(J[i,:]*state*dspin)
    
    return dH

#function that verifies the correct implementation of the energy_change() function

def cross_check_energy_change():

    print("Test: cross check of energy_change():")

    N = 20
    
    J = create_random_couplings(N)
    
    h = create_random_field(N)

    q = 0.5

    states = [create_state(q,N) for i in range(10)]

    success = True

    for state in states:

        site = random.randint(0,N-1)

        dH = energy_change(state,site,N,J,h)

        energy_old = energy(state, N, J, h)

        state[site] = -state[site]

        energy_new = energy(state, N, J, h)

        if (energy_new-energy_old) != dH:

            print("\t energy_new: ", energy_new)
            print("\t energy_old: ", energy_old)
            print("\t energy_diff: ",energy_new-energy_old)
            print("\t dH: ", dH)

            success = False
    
    print("\t success: ", success)

# implements one entire monte carlo sweep

@jit(nopython=True)   
def metropolis_procedure(state,N, J, h, number_sweeps):

    new_state = np.copy(state)
    
    # one monte-carlo sweep

    for i in range(number_sweeps):
        
        # pick random site
        
        rd_site = np.random.randint(0,N)

        # calculate energy change
        
        dH = energy_change(new_state,rd_site, J ,h)

        # if energy change is negative, flip spin
        # if not, draw a random number, and then maybe flip the spin
        
        if(dH>0):
            
            #draw random number
            p = np.random.random_sample()
            # if p < e^-dH/T, then flip the spin
            if(p<np.exp(-dH)):
                #flip the spin
                new_state[rd_site]=-1*new_state[rd_site]
                
        else:
            new_state[rd_site]=-1*new_state[rd_site]

    return new_state

# implements the metropolis algorithm for an input temperature T, system size L
# after burn-in time t_0, sampling starts
# measurement of m,e,m^2 and e^2

def metropolis(state, N, M, J, h, number_sweeps, burn_in_time = 1000):
    
    energy_obs = []
    
    states = []

    correlation = np.zeros((N,N))
    
    # burn in phase
    for t in range(0,burn_in_time):
            
        state = metropolis_procedure(state,N, J, h , number_sweeps)

    print("burn-in finished")
    
    # sampling starts
    
    for t in range(0,M):
        
        # do one monte carlo sweep
        
        state = metropolis_procedure(state, N, J, h, number_sweeps)

        correlation += (1/M)*np.tensordot(state, state, axes = 0)

        states.append(state)
        
        E = energy(state,N, J, h)
        
        energy_obs.append(E/N)
    
    # calc the average of the sums

    return np.mean(energy_obs), np.mean(states, axis = 0), correlation, states


# calculates the log likelihood for a given set of spin samples

def calc_log_likelihood_fast(spin_samples):

    M = len(spin_samples)
    
    # np.unique returns a dictionary with the counts
    
    values, counts = np.unique(spin_samples, axis = 0, return_counts = True)

    prob = counts/M

    log_likelihood = -(1/M)*np.sum(np.log(prob))

    return log_likelihood

# slower version, that also calculates the log likelihood

def calc_log_likelihood(spin_samples):

    # calculate prob distr. P_theta(s^k)
    # given by counts of s^k / M

    M = len(spin_samples)

    prob_distr = calc_prob_distr_from_samples(spin_samples, M)

    terms = []

    for key in prob_distr:

        term = -(1/M)*np.log(prob_distr[key])
        terms.append(term)

    return np.sum(terms)

# creates a dictionary from a given set of spin samples

def calc_prob_distr_from_samples(samples, M):

    prob_distr = {}

    for sample in samples:

        key = np.array2string(sample)

        if key in prob_distr:

            prob_distr[key] += 1/M

        else:

            prob_distr[key] = 1/M
    
    return prob_distr

# boltzmann learning algorithm minimizing the log-likelihood by gradient descent

def boltzmann_learning_algorithm(mean_spin_data, mean_correlation_data, learning_rate, N, steps_metropolis, iter_boltzmann, number_sweeps, tol = 0.001):
    
    # create random fields
    
    h = create_random_field(N)
    J = create_random_couplings(N)

    iterations_h = np.array([h])
    iterations_J = np.array([J])
    iterations_llhood = np.array([])

    for i in range(iter_boltzmann):

        print("starting with step ",i)
        print("starting metropolis")
        
        # do a metropolis simulation with the current model parameters

        mean_energy, mean_spin, mean_correlation, states = metropolis(create_state(0.5, N), N, steps_metropolis, J, h, number_sweeps)

        print("finished metropolis")
        
        h_old = h
        J_old = J

        # adapt model parameters

        derivative_h = (mean_spin_data-mean_spin)
        derivative_J = (mean_correlation_data-mean_correlation)
        
        
        max_h = np.amax(np.abs(derivative_h))
        max_J = np.amax(np.abs(derivative_J))

        delta_h = learning_rate * derivative_h
        delta_J = learning_rate * derivative_J

        print("mean spin data: ", mean_spin_data)
        print("mean spin model: ", mean_spin)
        print("delta h: ", delta_h)

        print("mean correlation data: ")
        print(mean_correlation_data)
        print("mean correlation model: ")
        print(mean_correlation)
        print("delta J: ")
        print(delta_J)

        np.fill_diagonal(derivative_J,0) # it is not possible for a spin to couple with itself

        iterations_h = np.concatenate((iterations_h, [derivative_h]), axis = 0)
        iterations_J = np.concatenate((iterations_J, [derivative_J]), axis = 0)
        
        # actually adapt model parameters
        
        h = h + delta_h
        J = J + delta_J
        
        # calculate the distance from old to new
        
        dist_h = np.linalg.norm(h-h_old)
        dist_J = np.linalg.norm(J-J_old)

        iterations_h = np.concatenate((iterations_h, [h]), axis = 0)
        iterations_J = np.concatenate((iterations_J, [J]), axis = 0)
        
        # calculate log-likelihood
        
        log_likelihood = calc_log_likelihood_fast(states)

        iterations_llhood = np.append(iterations_llhood, log_likelihood)

        print("step ", i,"; dist. h: ", dist_h, "; dist. J: ", dist_J, "; log-likelihood: ",log_likelihood)
        
        # some kind of convergence condition that has not been fulfilled a single time during my simulations
        
        if (max_h < tol) & (max_J < tol):

            print("converged")
            break

    
    return iterations_h, iterations_J, iterations_llhood

#tests if the correlation in the metropolis function is calculated correctly

def test_correlation_of_metropolis():

    N = 5

    h = create_random_field(N)
    J = create_random_couplings(N)

    energy, magnetisation, correlation, states = metropolis(create_state(0.5,N), N, 100000, J, h, N*N)

    correlation_test = []

    for state in states:

        correlation_test.append(calc_tensor_product(state, state))
    
    correlation_test = np.mean(correlation_test, axis = 0)

    print(correlation)
    print(correlation_test)

    dist = np.linalg.norm(correlation-correlation_test)

    if dist == 0:
        print("Metropolis calculates the correlation correctly.")
    else:
        print(dist)

# calculates the tensor product, never used, just to test if np.tensordot does the correct thing

def calc_tensor_product(state_a, state_b):
    N = len(state_a)
    correlation = np.zeros((N,N))
    
    for i in range(N):
        
        for j in range(N):
            
            correlation[i,j] = state_a[i]*state_b[j]
            
    return correlation

# calculates the correlation of spin triplets

def calc_correlation_triplets(states, N, cut_off = 10):
    
    M = len(states)
    
    correlation = np.zeros((cut_off, cut_off, cut_off))
    
    i = 0
    
    for state in states:
        
        if(i%100)==0:
            print("step ", i)
        
        #first tensorproduct
        
        two_spin_correlation = np.tensordot(state[0:cut_off], state[0:cut_off], axes = 0)
        
        #second tensorproduct
        
        three_spin_correlation = np.tensordot(state[0:cut_off], two_spin_correlation, axes = 0)
        correlation += (1/M)*three_spin_correlation
        
        i = i + 1
    
    return correlation, cut_off

# a method that tries to infer target parameters h and J for a given learning rate

def infer_parameters(task, target_h, target_J,learning_rate, steps_training_data, iter_boltzmann, N):

    steps_metropolis = 100000
    
    #calculate mean spin and mean correlation of the target parameters

    mean_energy, spin_training_data, correlation_training_data, states = metropolis(create_state(0.5, N), N, steps_training_data, target_J, target_h, N*N)
    
    # try to infer the field and couplings
    
    iterations_h, iterations_J, iterations_llhood = boltzmann_learning_algorithm(spin_training_data, correlation_training_data, learning_rate, N, steps_metropolis, iter_boltzmann, N*N)
    
    # dump results in a file
    
    dump_iterations(task, target_h, target_J, iterations_h, iterations_J, iterations_llhood, N, steps_training_data, steps_metropolis, learning_rate, iter_boltzmann)

# method that dumps the result into a file

def dump_iterations(task, target_h, target_J, iterations_h, iterations_J, iterations_llhood, N, steps_training_data, steps_metropolis, learning_rate, iter_boltzmann):
    
    iterations_h = np.insert(iterations_h, 0, target_h, axis = 0)
    h_array = np.reshape(iterations_h,(-1,N))

    np.savetxt("data/" + task + "_iterations_h_eta=" + str(learning_rate).replace(".","") + "_tsteps=" + str(steps_training_data) +".txt", h_array, header = "N = " + str(N) + "; steps_train_data = " + str(steps_training_data) + "; steps_metropolis = " + str(steps_metropolis) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iter_boltzmann))

    iterations_J = np.insert(np.reshape(iterations_J,(-1,N*N)), 0, np.reshape(target_J,(-1,N*N)), axis = 0)
    #J_array = np.reshape(iterations_J,(-1,N*N))
    
    np.savetxt("data/" + task +"_iterations_J_eta=" + str(learning_rate).replace(".","") + "_tdata=" + str(steps_training_data) +".txt", iterations_J, header = "N = " + str(N) + "; steps_train_data = " + str(steps_training_data) + "; steps_metropolis = " + str(steps_metropolis) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iter_boltzmann))

    np.savetxt("data/" + task + "_log_likelihood_eta=" + str(learning_rate).replace(".","") +"_tdata=" + str(steps_training_data) +".txt", iterations_llhood, header = "N = " + str(N) + "; steps_train_data = " + str(steps_training_data) + "; steps_metropolis = " + str(steps_metropolis) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iter_boltzmann))

# method that dumps the result of the salamander data into a file

def dump_iterations_salamander(task, iterations_h, iterations_J, iterations_llhood, N, steps_metropolis, learning_rate, iter_boltzmann):
    
    h_array = np.reshape(iterations_h,(-1,N))

    np.savetxt("data_salamander/" + task +"_iterations_h_eta=" + str(learning_rate).replace(".","") + ".txt", h_array, header = "N = " + str(N) + "; steps_train_data = " + "; steps_metropolis = " + str(steps_metropolis) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iter_boltzmann))

    iterations_J = np.reshape(iterations_J,(-1,N*N))
    
    np.savetxt("data_salamander/" + task + "_iterations_J_eta=" + str(learning_rate).replace(".","") + ".txt", iterations_J, header = "N = " + str(N) + "; steps_metropolis = " + str(steps_metropolis) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iter_boltzmann))

    np.savetxt("data_salamander/" + task + "_log_likelihood_eta=" + str(learning_rate).replace(".","")  + ".txt", iterations_llhood, header = "N = " + str(N) + "; steps_train_data = " + "; steps_metropolis = " + str(steps_metropolis) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iter_boltzmann))

# method that calculates the magnetisation and the mean correlation from the salamander data

def calc_magnetisation_correlation_from_salamander_activity(data):

    # transformation

    data = (data-0.5)*2

    mean_magnetisation = np.mean(data, axis = 1)

    M = len(data[0,:])
    N = 160
    
    correlation = np.zeros((N,N))

    for i in range(M-1):
        
        #correlation-matrix M_ij = s_i(t)s_j(t)
        correlation += (1/(M-1))*(np.tensordot(data[:,i], data[:,i], axes = 0))

    return mean_magnetisation, correlation


def task_3_1_4():

    N = 5

    target_h = create_random_field(N)
    target_J = create_random_couplings(N)

    infer_parameters("3_1_4", target_h, target_J, 0.01, 100000, 150, N)
    infer_parameters("3_1_4", target_h, target_J, 0.05, 100000, 150, N)
    infer_parameters("3_1_4", target_h, target_J, 0.10, 100000, 150, N)

def task_3_1_5():

    N = 5

    target_h = create_random_field(N)
    target_J = create_random_couplings(N)

    infer_parameters("3_1_5", target_h, target_J, 0.05, 50000, 200, N)
    infer_parameters("3_1_5", target_h, target_J, 0.05, 100000, 200, N)
    infer_parameters("3_1_5", target_h, target_J, 0.05, 500000, 200, N)
    infer_parameters("3_1_5", target_h, target_J, 0.05, 1000000, 200, N)

def task_3_2_1_1():

    N = 160
    learning_rate = 0.05
    steps_metropolis = 100000
    iter_boltzmann = 200

    data = np.loadtxt("bint.txt")

    spin_training_data, correlation_training_data = calc_magnetisation_correlation_from_salamander_activity(data)

    iterations_h, iterations_J, iterations_llhood = boltzmann_learning_algorithm(spin_training_data, correlation_training_data, learning_rate, N, steps_metropolis, iter_boltzmann, N)

    dump_iterations_salamander("data_3_2_1_1", iterations_h, iterations_J, iterations_llhood, 160, steps_metropolis, learning_rate, iter_boltzmann)

def task_3_2_1_2():
    
    N = 160
    learning_rate = 0.05
    steps_metropolis = 100000
    iter_boltzmann = 200

    data = np.loadtxt("bint.txt")
    length = len(data[0,:])

    results_spin = []
    results_correlation = []

    # train model only on the first half of the dataset

    training_data = data[:,0:int(length/2)]

    # calculate training data

    spin_training_data, correlation_training_data = calc_magnetisation_correlation_from_salamander_activity(training_data)

    results_spin.append(spin_training_data)
    results_correlation.append(np.reshape(correlation_training_data,(-1,N*N)))

    # train model with training data

    iterations_h, iterations_J, iterations_llhood = boltzmann_learning_algorithm(spin_training_data, correlation_training_data, learning_rate, N, steps_metropolis, iter_boltzmann, N)

    h = iterations_h[-1]
    J = iterations_J[-1]

    dump_iterations_salamander("data_3_2_1_2_a", iterations_h, iterations_J, iterations_llhood, 160, steps_metropolis, learning_rate, iter_boltzmann)
    
    # calculate mean spin and correlation for the obtained h and J
    
    mean_energy, mean_spin, mean_correlation, states = metropolis(create_state(0.5, N), N, steps_metropolis, J, h, N)
    
    # append it to array
    
    results_spin.append(mean_spin)
    results_correlation.append(np.reshape(mean_correlation,(-1,N*N)))
    
    # calculate the checking data set
    
    checking_data = data[:,int(length/2):]

    spin_checking_data, correlation_checking_data = calc_magnetisation_correlation_from_salamander_activity(checking_data)

    results_spin.append(spin_checking_data)
    results_correlation.append(np.reshape(correlation_checking_data, (-1,N*N)))
    
    # calculate the difference

    distance_spin= np.linalg.norm(spin_checking_data-mean_spin)/N
    distance_correlation = np.linalg.norm(correlation_checking_data-mean_correlation)/(N*N)
    
    # save the data
    
    results_correlation = np.reshape(results_correlation,(3,N*N))

    np.savetxt("data_salamander/data_3_2_1_2_a_results_spin", results_spin, header = "#1: training_data; #2: model_data; #3: checking_data; distance_spin =" + str(np.round(distance_spin,10)) + "; distance_correlation =" + str(np.round(distance_correlation,6)) + ";")
    np.savetxt("data_salamander/data_3_2_1_2_a_results_correlation", results_correlation, header = "#1: training_data; #2: model_data; #3: checking_data; distance_spin =" + str(np.round(distance_spin,10)) + "; distance_correlation =" + str(np.round(distance_correlation,6)) + ";")

    results_spin = []
    results_correlation = []

    # train model on every second dataset

    training_data = data[:,0::2]

    spin_training_data, correlation_training_data = calc_magnetisation_correlation_from_salamander_activity(training_data)

    results_spin.append(spin_training_data)
    results_correlation.append(np.reshape(correlation_training_data,(-1,N*N)))

    # train model with training data

    iterations_h, iterations_J, iterations_llhood = boltzmann_learning_algorithm(spin_training_data, correlation_training_data, learning_rate, N, steps_metropolis, iter_boltzmann, N)

    h = iterations_h[-1]
    J = iterations_J[-1]

    dump_iterations_salamander("data_3_2_1_2_b", iterations_h, iterations_J, iterations_llhood, 160, steps_metropolis, learning_rate, iter_boltzmann)
    
    # calculate mean spin and correlation for the obtained h and J

    mean_energy, mean_spin, mean_correlation, states = metropolis(create_state(0.5, N), N, steps_metropolis, J, h, N)
    
    # append it to array
    
    results_spin.append(mean_spin)
    results_correlation.append(np.reshape(mean_correlation, (-1,N*N)))
    
    # calculate the checking data set

    checking_data = data[:,1::2]

    spin_checking_data, correlation_checking_data = calc_magnetisation_correlation_from_salamander_activity(checking_data)

    results_spin.append(spin_checking_data)
    results_correlation.append(np.reshape(correlation_checking_data,(-1,N*N)))
    
    # calculate the difference

    distance_spin= np.linalg.norm(spin_checking_data-mean_spin)/N
    distance_correlation = np.linalg.norm(correlation_checking_data-mean_correlation)/(N*N)

    results_correlation = np.reshape(results_correlation,(3,N*N))
    
    np.savetxt("data_salamander/data_3_2_1_2_b_results_spin", results_spin, header = "#1: training_data; #2: model_data; #3: checking_data; distance_spin =" + str(np.round(distance_spin,10)) + "; distance_correlation =" + str(np.round(distance_correlation,6)) + ";")
    np.savetxt("data_salamander/data_3_2_1_2_b_results_correlation", results_correlation, header = "#1: training_data; #2: model_data; #3: checking_data; distance_spin =" + str(np.round(distance_spin,10)) + "; distance_correlation =" + str(np.round(distance_correlation,6)) + ";")
    
def task_3_2_1_4():
    
    N = 160
    steps_metropolis = 300000
    
    # pull the most precise h and J from the data
    
    data_h = np.loadtxt("data_salamander/data_3_2_1_1_iterations_h_eta=005.txt")
    data_J = np.loadtxt("data_salamander/data_3_2_1_1_iterations_J_eta=005.txt")
    
    h = data_h[-1,:]
    J = data_J[-1,:]
    
    J = np.reshape( J, (N,N))
     
    print(len(h))
    print(np.shape(J))
    
    data = np.loadtxt("bint.txt")
    
    transpose_data = np.transpose(data)
    
    print("transposing finished")
    
    # calculate spin triplet correlation from the data
    
    correlation, cut_off = calc_correlation_triplets(transpose_data, N)
    
    # do a metropolis simulation with the most precise model parameters
    
    mean_energy, mean_spin, mean_correlation, states = metropolis(create_state(0.5, N), N, steps_metropolis, J, h, N)
    
    # calculate the spin triplet correlation
    
    correlation_model, cut_off = calc_correlation_triplets(states, N)
    
    # calculate the distance
    
    dist = np.linalg.norm(correlation-correlation_model)/(cut_off**3)
    
    np.savetxt("data_salamander/data_3_2_1_4_three_spin_correlation.txt", [correlation.flatten(), correlation_model.flatten()], header = "#1 training data; #2 model data; diff = " + str(np.round(dist, 8)) + "; cut_off = " + str(cut_off))


task_3_2_1_4()