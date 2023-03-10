#!<path_to_GERDA_virtuel_environment>/envs/gerdaenv/bin/python3.8
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:32:35 2022

This script computes in parallel Monte Carlo estimates of the state and the
corresponding objective function for a given control. The inputs for this
script are automatically loaded from the data_path specified in
opt_settings.txt. The results are saved in the same directory.

@author: Jan-Hendrik Niemann
"""


import numpy as np
import os
import json
import time
from joblib import Parallel, delayed
import multiprocessing

from GERDA_aux import objective, stretch_vec
from GERDA_dummy_ABM import markov_jump_process


# Path to settings file
path = '<path_to_input_and_output_directory>/opt_settings.txt'

tic = time.time()

# Open json file
with open(path, 'r') as file:
    settings_dict = json.load(file)

# Load general settings
total_timesteps = settings_dict['total_timesteps']
duration_of_constant_control = settings_dict['duration_of_constant_control']
data_path = settings_dict['data_path']
output_dir = settings_dict['output_directory']
T_max = (settings_dict['total_timesteps'] - 1) / 24
t_step = 1 / 24
c1 = settings_dict['c1']
samplesize_max = settings_dict['samplesize_max']

# Initial aggregate state
y0 = np.genfromtxt(os.path.join(data_path, output_dir, 'y0.csv'), delimiter=',', dtype=int)
x_init = np.zeros_like(y0)
x_init[0] = y0[0]  # Adults susceptible
x_init[1] = y0[2]  # Adults infected
x_init[2] = y0[4]  # Adults recovered
x_init[3] = y0[1]  # Children susceptible
x_init[4] = y0[3]  # Children infected
x_init[5] = y0[5]  # Children recovered

# Parameter of GERDA fitted model
p = np.genfromtxt(os.path.join(data_path, output_dir, 'fitting_parameter.csv'), delimiter=',')

# Load control vector
U = np.genfromtxt(os.path.join(data_path, 'control_U.csv'), delimiter=',')
if U.ndim == 1:
    U = np.expand_dims(U, 0)
Us = U[:, 0]
Uw = U[:, 1]

# Number of Monte Carlo simulations
num_MC_sim = np.genfromtxt(os.path.join(data_path, 'num_MC_sim_obj_est.csv'), delimiter=',', dtype=int).item()

# Number of available cores for parallelization
num_cores = multiprocessing.cpu_count()

# Limit number of cores used
if num_MC_sim < num_cores:
    num_cores = num_MC_sim

# Load projected gradient estimate
try:
    Ju = np.genfromtxt(os.path.join(data_path, 'gradient_estimate_projected.csv'), delimiter=',')
    Ju = Ju.flatten(order='F')
except OSError:
    Ju = np.zeros_like(U)
    Ju = Ju.flatten(order='F')

# Current step size
try:
    alpha = np.genfromtxt(os.path.join(data_path, 'alpha.csv'), delimiter=',')
except OSError:
    alpha = 1

# Minimum relative accuracy gradient estimate
try:
    eps = np.genfromtxt(os.path.join(data_path, 'eps.csv'), delimiter=',')
except OSError:
    eps = settings_dict['eps']

# Error bound e
try:
    error_e = np.genfromtxt(os.path.join(data_path, 'error_e.csv'), delimiter=',')
except OSError:
    error_e = eps * c1 * alpha * np.linalg.norm(Ju, ord=2)**2

# %% Execute

# Create empty array
ABM_samples = np.zeros((0, 6, total_timesteps))

# Parallel execution
with Parallel(n_jobs=num_cores, verbose=0) as parallel:
    while True:

        samplesize_add = int(num_MC_sim * 0.5)

        # Random variable for better seed
        rnd_var = np.random.randint(0, 2**32 - 1)

        # If no samples are done yet, create initial samples
        if ABM_samples.shape[0] == 0:
            result = parallel(delayed(markov_jump_process)(x_init, p, U, t_step, T_max, settings_dict, sim_id + rnd_var) for sim_id in range(num_MC_sim))

            ABM_samples_add = np.asarray(result, dtype=float)

        # If accuracy is too low, create more samples
        else:
            result = parallel(delayed(markov_jump_process)(x_init, p, U, t_step, T_max, settings_dict, sim_id + rnd_var) for sim_id in range(samplesize_add))
            ABM_samples_add = np.asarray(result, dtype=float)

        # Concatenate samples
        ABM_samples = np.concatenate((ABM_samples, ABM_samples_add), axis=0)

        # Update number of samples
        num_MC_sim = ABM_samples.shape[0]

        # Get results
        J_vec = np.zeros(num_MC_sim)

        for i in range(num_MC_sim):
            SIR_all = ABM_samples[i, :, :]

            adults = SIR_all[(0, 1, 2), :].T
            children = SIR_all[(3, 4, 5), :].T

            J_vec[i] = objective(adults, children,
                                 stretch_vec(Us, Uw, total_timesteps, duration_of_constant_control), settings_dict)

        # Sample Mean
        J_mean = np.mean(J_vec)

        # Standard deviation of sample mean
        J_std = np.std(J_vec, ddof=1) / np.sqrt(num_MC_sim)

        # Test accuracy
        if 2 * J_std <= error_e:
            print('Accuracy reached with error e = %g for %g Monte Carlo simulations' % (2 * J_std, num_MC_sim))
            break
        elif error_e <= 0:
            print('Accuracy cannot be checked...\n|Ju| = %g, eps = %g, alpha = %g, c1 = %g' % (np.linalg.norm(Ju, ord=2), eps, alpha, c1))
            print('Current error e = %g' % (2 * J_std))
            break
        elif num_MC_sim >= samplesize_max:
            print('Maximum number of Monte Carlo simulations reached')
            print('Current error e = %g' % (2 * J_std))
            break
        else:
            print('Error e = %g !< %g' % (2 * J_std, error_e))

# Sample mean
ABM_sample_mean = np.mean(ABM_samples, axis=0)
state_adults_mean = ABM_sample_mean[(0, 1, 2), :]
state_children_mean = ABM_sample_mean[(3, 4, 5), :]

# Variance
J_sample_var = np.var(ABM_samples, axis=0, ddof=1)
state_adults_var = J_sample_var[(0, 1, 2), :]
state_children_var = J_sample_var[(3, 4, 5), :]

# Standard deviation
J_sample_std = np.std(ABM_samples, axis=0, ddof=1)
state_adults_std = J_sample_std[(0, 1, 2), :]
state_children_std = J_sample_std[(3, 4, 5), :]

# %% Save tmp to csv files

# Save tmp files as csv
np.savetxt(os.path.join(data_path, output_dir, 'adults_state_estimate_mean_tmp.csv'), state_adults_mean.T, delimiter=',')
np.savetxt(os.path.join(data_path, output_dir, 'children_state_estimate_mean_tmp.csv'), state_children_mean.T, delimiter=',')
np.savetxt(os.path.join(data_path, output_dir, 'adults_state_estimate_var_tmp.csv'), state_adults_var.T, delimiter=',')
np.savetxt(os.path.join(data_path, output_dir, 'children_state_estimate_var_tmp.csv'), state_children_var.T, delimiter=',')
np.savetxt(os.path.join(data_path, output_dir, 'adults_state_estimate_std_tmp.csv'), state_adults_std.T, delimiter=',')
np.savetxt(os.path.join(data_path, output_dir, 'children_state_estimate_std_tmp.csv'), state_children_std.T, delimiter=',')

with open(data_path + '/objective_estimate.csv', 'w') as file:
    file.write(str(J_mean))

with open(data_path + '/objective_sample_mean_std.csv', 'w') as file:
    file.write(str(J_std))

with open(os.path.join(data_path, 'num_MC_sim_obj_est.csv'), 'w') as file:
    file.write(str(num_MC_sim))

print('Elapsed time %g minutes\n' % ((time.time() - tic) / 60))

# %% Save auxiliray output and samples

# Create dictionary with settings
out_dict = {"e_forced": float(error_e),
            "e_achived": float(2 * J_std),
            "samplesize": num_MC_sim,
            "objective_mean": J_mean,
            "sample_mean_standard_deviation": J_std
            }

counter = 0
while os.path.exists(os.path.join(data_path, output_dir, 'Ju_aux_out_' + str(counter) + '.json')):
    counter += 1

# Save as json encoded file
with open(os.path.join(data_path, output_dir, 'J_aux_out_' + str(counter) + '.json'), 'w') as json_file:
    json.dump(out_dict, json_file, indent=4, sort_keys=True, ensure_ascii=True)

np.savez_compressed(os.path.join(data_path, output_dir, 'Samples_' + str(counter) + '.npz'), adults=adults, children=children, J_vec=J_vec)
