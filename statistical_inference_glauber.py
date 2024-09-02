import numpy as np
import math
import random

import matplotlib.pyplot as plt
import os

from numba import jit

# creates a state, where a fraction of q spins point down

@jit(nopython=True)
def create_state(q,N):
    
    state = np.ones(N)
    
    for i in range(0,math.floor(q*N)):
        
        state[i]=-1.0
        
    np.random.shuffle(state)
    
    return state

# creates a random field h

@jit(nopython=True)
def create_random_field(N):

    field = np.random.standard_normal(N)
    field = field/np.sqrt(N)

    return field

# creates a random coupling (not symmetric)

@jit(nopython=True)
def create_random_couplings(N):
    
    couplings = np.random.standard_normal((N,N))/np.sqrt(N)
    
    np.fill_diagonal(couplings,0)

    return couplings

# creates a symmetric random coupling

@jit(nopython=True)
def create_random_symmetric_couplings(N):

    couplings = np.zeros((N,N))

    for i in range(N):

        for j in range(N):

            if i > j:

                couplings[i,j] = couplings[j,i]
            
            else:

                couplings[i,j] = np.random.normal(loc = 0, scale = 1/np.sqrt(N)) 
    
    np.fill_diagonal(couplings,0)

    return couplings

# tests if mean and variance of the couplings are produced correctly

def test_randomness_of_couplings():

    couplings = []

    for i in range(100000):

        couplings.append(create_random_couplings(10))
    
    mean = np.mean(couplings, axis = 0)
    var = np.mean(np.power(couplings,2), axis = 0)

    return mean, var

# calculates the mean of a series of states

def calc_mean_time_series(states, N):
    
    # calc mean
    mean_spin = np.mean(states, axis = 0)
    
    return mean_spin

# calculates the temporal correlations for a series of states

def calc_temporal_correlation(states, N):
    
    M = len(states)
    
    correlation = np.zeros((N,N))

    for i in range(len(states)-1):
        
        # calculate the tensor product s(t+1)s(t)
        
        correlation += (1/(M-1))*(np.tensordot(states[i+1], states[i], axes = 0))
    
    return correlation


# calculates the derivative of the log-likelihood

def calc_derivates_log(states, h, J, N):
    
    M = len(states)
    
    dev_h = np.zeros(N)
    dev_J = np.zeros((N,N))
    
    for i in range(M-1):
        
        #calculating theta
        
        theta = calc_theta(states[i], h, J)
        
        # derivative in h
        
        dev_h += (1/M)*(states[i+1]-np.tanh(theta))
        
        # calculating correlation
        
        correlation = np.tensordot(states[i+1], states[i], axes = 0)
        
        # calculating <theta s(t)>
        
        other_term = np.tensordot(np.tanh(theta), states[i], axes = 0)
        
        dev_J += (1/M)*(correlation-other_term)
    
    return dev_h, dev_J

# implementation of the glauber dynamics

@jit(nopython=True)
def glauber_dynamics(state, J, h, N):

    new_state = np.copy(state)

    theta = calc_theta(state,h, J)
    
    # calculating the different transition probabilities
    
    p = np.exp(-1*np.multiply(state,theta))/(2*np.cosh(theta))

    for i in range(N):

        prob_flip = np.random.random_sample()
        
        # flipping spins
        
        if prob_flip <= p[i]:

            new_state[i] = new_state[i]*(-1)
    
    return new_state

# generate time series using glauber dynamics

def generate_time_series(N, J, h, n_samples):

    state = create_state(0.5,N)

    states = []
    
    for i in range(n_samples):

        state = glauber_dynamics(state, J, h, N)
        
        states.append(state)

    return states

# some test function to see if glauber dynamics work correctly
# it compares <s> to <theta> and <s(t+1)s(t)> to < theta s(t)>, taken from the paper

def test_glauber_dynamics():
    
    N = 3
    
    J = create_random_couplings(N)
    h = create_random_field(N)
    
    states = generate_time_series(N, J, h, 100000)
    magnetisation = calc_mean_time_series(states, N)
    correlation = calc_temporal_correlation(states, N)
    theta = []
    theta_correlation = []
    
    for state in states:
        
        tanh_theta = np.tanh(calc_theta(state, h, J))
        theta.append(tanh_theta)
        theta_correlation.append(np.tensordot(tanh_theta, state, axes = 0))
        
    mean_theta = np.mean(theta, axis = 0)
    mean_theta_correlation = np.mean(theta_correlation, axis = 0)
    
    print("mean magnetisation: ", magnetisation)
    print("mean theta: ",  mean_theta) 
    
    print("correlation: ")
    print(correlation)
    
    print("correlation theta: ")
    print(mean_theta_correlation)  
    
    
# function that saves the time series to a file

def save_time_series(states, N):

    np.savetxt("time_series_data/time_series.txt", states, header = "N = " + str(N) +"; n_samples = " + str(len(states)) + ";")

# calculates the log-likelihood

def calc_log_likelihood(time_series_data, J, h):

    M = len(time_series_data)

    log_likelihood = 0
    
    #iterating over the sample
    
    for j in range(M-1):

        theta = calc_theta(time_series_data[j], h, J)

        term = time_series_data[j+1]*theta-np.log(2*np.cosh(theta))

        log_likelihood += (1/M)*np.sum(term)
    
    return log_likelihood

# calculates theta

@jit(nopython=True)
def calc_theta(state, h, J):
    
    return np.asarray([np.sum(J[i,:]*state)+h[i] for i in range(len(state))])

def test_calc_theta():
    
    N = 3
    
    h = np.ones(N)
    J = np.zeros((N,N))
    J[1,0] = 1
    J[1,1] = 0
    J[1,2] = 2
    
    state = np.arange(1,4,1)
    
    print(calc_theta(state, h, J))

# test function to compare np.tensordot against to see if I implemented it correctly

def calc_tensor_product(state_a, state_b):
    N = len(state_a)
    correlation = np.zeros((N,N))
    
    for i in range(N):
        
        for j in range(N):
            
            correlation[i,j] = state_a[i]*state_b[j]
            
    return correlation

# actual learning algorithm

def learning_algorithm(states, learning_rate, N, iterations, symmetric = False):
    
    # generate couplings
    
    J = create_random_couplings(N)
    
    if symmetric:
        
        J = create_random_symmetric_couplings(N)
        
    h = create_random_field(N)
    
    # setting up arrays to save it to
    
    iterations_h = [h]
    iterations_J = [J]
    iterations_llhood = []

    for i in range(iterations):

        print("starting with step ",i)
        
        # calculating the derivatives
        
        derivative_h, derivative_J = calc_derivates_log(states, h, J, N)
        
        np.fill_diagonal(derivative_J,0) # enforce that no coupling to itself is allowed
        
        if symmetric:
            
            derivative_J = 0.5*(derivative_J+np.transpose(derivative_J))
            
        
        iterations_h.append(derivative_h)
        iterations_J.append(derivative_J)
        
        h_old = h
        J_old = J

        # adapt model parameters

        h = h + learning_rate * derivative_h
        J = J + learning_rate * derivative_J

        iterations_h.append(h)
        iterations_J.append(J)
        
        # calculating log_likelihood
        
        log_likelihood = calc_log_likelihood(states, J , h)

        iterations_llhood.append(log_likelihood)
        
        #some kind of distance measure to see if the iteration converges
        
        dist_h = np.linalg.norm(h-h_old)
        dist_J = np.linalg.norm(J-J_old)

        print("step ", i,"; dist. h: ", dist_h, "; dist. J: ", dist_J, "; log-likelihood: ",log_likelihood)
    
    return iterations_h, iterations_J, iterations_llhood

# some function to save the iterations to a file

def dump_iterations(task, target_h, target_J, iterations_h, iterations_J, iterations_llhood, N, steps_training_data, learning_rate, iterations):
    
    iterations_h = np.insert(iterations_h, 0, target_h, axis = 0)
    h_array = np.reshape(iterations_h,(-1,N))

    np.savetxt("data/" + task +"_iterations_h_eta=" + str(learning_rate).replace(".","") + "_tdata=" + str(steps_training_data) +".txt", h_array, header = "N = " + str(N) + "; steps_train_data = " + str(steps_training_data) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iterations))

    iterations_J = np.insert(np.reshape(iterations_J,(-1,N*N)), 0, np.reshape(target_J,(-1,N*N)), axis = 0)
    #J_array = np.reshape(iterations_J,(-1,N*N))
    
    np.savetxt("data/" + task +"_iterations_J_eta=" + str(learning_rate).replace(".","") + "_tdata=" + str(steps_training_data) +".txt", iterations_J, header = "N = " + str(N) + "; steps_train_data = " + str(steps_training_data) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iterations))

    np.savetxt("data/" + task + "_log_likelihood_eta=" + str(learning_rate).replace(".","") +"_tdata=" + str(steps_training_data) +".txt", iterations_llhood, header = "N = " + str(N) + "; steps_train_data = " + str(steps_training_data) + "; learning_rate = " + str(learning_rate) + "; iterations_boltzmann = " + str(iterations))

# some function to save the iterations to a file

def dump_iterations_salamander(task, iterations_h, iterations_J, iterations_llhood, N, learning_rate, iterations):
    
    h_array = np.reshape(iterations_h,(-1,N))

    np.savetxt("data/" + task +"_iterations_h_eta=" + str(learning_rate).replace(".","") + ".txt", h_array, header = "N = " + str(N) + "; learning_rate = " + str(learning_rate) + "; iterations = " + str(iterations))

    iterations_J = np.reshape(iterations_J,(-1,N*N))
    
    np.savetxt("data/" + task + "_iterations_J_eta=" + str(learning_rate).replace(".","") + ".txt", iterations_J, header = "N = " + str(N) + "; learning_rate = " + str(learning_rate) + "; iterations = " + str(iterations))

    np.savetxt("data/" + task + "_log_likelihood_eta=" + str(learning_rate).replace(".","")  + ".txt", iterations_llhood, header = "N = " + str(N) + "; iterations = " + str(iterations))

def task_3_2_2_2():
    
    #generate couplings and time series data 
    
    N = 5
    J = create_random_couplings(N)
    h = create_random_field(N)
    
    learning_rate = 0.1
    steps_training_data = 1000000
    iterations = 150
    
    states = generate_time_series(N, J, h, steps_training_data)
    
    save_time_series(states, N)
    
    iterations_h, iterations_J, iterations_llhood = learning_algorithm(states, learning_rate, N, iterations)
    
    dump_iterations("3_2_2_2",h, J, iterations_h, iterations_J, iterations_llhood, N, steps_training_data, learning_rate, iterations)
    
def task_3_2_2_3():
    
    N = 160
    learning_rate = 0.05
    iterations = 200
    
    data = np.loadtxt("bint.txt")
    data = (data-0.5)*2 # transformation
    
    states = np.transpose(data)
    
    iterations_h, iterations_J, iterations_llhood = learning_algorithm(states, learning_rate, N, iterations, symmetric = True)
    
    dump_iterations_salamander("3_2_2_3", iterations_h, iterations_J, iterations_llhood, N, learning_rate, iterations)
    
    h = iterations_h[-1]
    J = iterations_J[-1]
    
    n_samples = 300000
    
    new_states = generate_time_series(N, J, h, n_samples)
    
    infered_temporal_correlation = calc_temporal_correlation(new_states, N)
    
    data_temporal_correlation = calc_temporal_correlation(states, N)
    
    infered_temporal_correlation = np.ndarray.flatten(infered_temporal_correlation)
    data_temporal_correlation = np.ndarray.flatten(data_temporal_correlation)
    
    np.savetxt("data/3_2_2_3_inf_temp_correlation.txt", infered_temporal_correlation)
    np.savetxt("data/3_2_2_3_data_temp_correlation.txt", data_temporal_correlation)
     
def task_3_2_2_4():
    
    N = 160
    learning_rate = 0.05
    iterations = 200
    
    data = np.loadtxt("bint.txt")
    data = (data-0.5)*2 # transformation
    
    states = np.transpose(data)
    
    iterations_h, iterations_J, iterations_llhood = learning_algorithm(states, learning_rate, N, iterations)
    
    dump_iterations_salamander("3_2_2_4", iterations_h, iterations_J, iterations_llhood, N, learning_rate, iterations)
    
    h = iterations_h[-1]
    J = iterations_J[-1]
    
    n_samples = 500000
    
    new_states = generate_time_series(N, J, h, n_samples)
    
    infered_temporal_correlation = calc_temporal_correlation(new_states, N)
    
    data_temporal_correlation = calc_temporal_correlation(states, N)
    
    infered_temporal_correlation = np.ndarray.flatten(infered_temporal_correlation)
    data_temporal_correlation = np.ndarray.flatten(data_temporal_correlation)
    
    np.savetxt("data/3_2_2_4_inf_temp_correlation.txt", infered_temporal_correlation)
    np.savetxt("data/3_2_2_4_data_temp_correlation.txt", data_temporal_correlation)

    
task_3_2_2_3()
task_3_2_2_4()
