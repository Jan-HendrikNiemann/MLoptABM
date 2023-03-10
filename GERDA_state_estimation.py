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
import copy
import time
from joblib import Parallel, delayed
import multiprocessing

from GERDA_aux import load, baseline_scenario, objective, stretch_vec


# Path to settings file
path = '<path_to_input_and_output_directory>/opt_settings.txt'

tic = time.time()

with open(path, 'r') as file:
    settings_dict = json.load(file)

# Load world
modeledWorld = load(settings_dict, create_new_world=True)

# Load settings
total_timesteps = settings_dict['total_timesteps']
duration_of_constant_control = settings_dict['duration_of_constant_control']
data_path = settings_dict['data_path']
output_dir = settings_dict['output_directory']
c1 = settings_dict['c1']
samplesize_max = settings_dict['samplesize_max']

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

# Create empty arrays
state_adults = np.zeros((0, 3, total_timesteps))
state_children = np.zeros((0, 3, total_timesteps))
J_vec = np.zeros(0)

# Parallel execution
with Parallel(n_jobs=num_cores, verbose=0) as parallel:
    while True:

        samplesize_add = int(num_MC_sim * 0.5)

        # Random variable for better seed
        rnd_var = np.random.randint(0, 2**32 - 1)

        # If no samples are done yet, create initial samples
        if J_vec.shape[0] == 0:
            result = parallel(delayed(baseline_scenario)(copy.deepcopy(modeledWorld), Us, Uw, settings_dict, sim_id + rnd_var) for sim_id in range(num_MC_sim))

            # Get results
            state_adults_add = np.zeros((num_MC_sim, 3, total_timesteps))
            state_children_add = np.zeros((num_MC_sim, 3, total_timesteps))
            J_vec_add = np.zeros(num_MC_sim)

            for i in range(num_MC_sim):
                adults, children = result[i]

                J = objective(adults, children,
                              stretch_vec(Us, Uw, total_timesteps, duration_of_constant_control), settings_dict)

                state_adults_add[i, 0, :] = adults.S
                state_adults_add[i, 1, :] = adults.I

                state_children_add[i, 0, :] = children.S
                state_children_add[i, 1, :] = children.I

                if adults.shape[1] == 4:
                    state_adults_add[i, 2, :] = (adults.R + adults.D)
                else:
                    state_adults_add[i, 2, :] = adults.R

                if children.shape[1] == 4:
                    state_children_add[i, 2, :] = (children.R + children.D)
                else:
                    state_children_add[i, 2, :] = children.R

                J_vec_add[i] = J

        # If accuracy is too low, create more samples
        else:
            result = parallel(delayed(baseline_scenario)(copy.deepcopy(modeledWorld), Us, Uw, settings_dict, sim_id + rnd_var) for sim_id in range(samplesize_add))

            # Get results
            state_adults_add = np.zeros((samplesize_add, 3, total_timesteps))
            state_children_add = np.zeros((samplesize_add, 3, total_timesteps))
            J_vec_add = np.zeros(samplesize_add)

            for i in range(samplesize_add):
                adults, children = result[i]

                J = objective(adults, children,
                              stretch_vec(Us, Uw, total_timesteps, duration_of_constant_control), settings_dict)

                state_adults_add[i, 0, :] = adults.S
                state_adults_add[i, 1, :] = adults.I

                state_children_add[i, 0, :] = children.S
                state_children_add[i, 1, :] = children.I

                if adults.shape[1] == 4:
                    state_adults_add[i, 2, :] = (adults.R + adults.D)
                else:
                    state_adults_add[i, 2, :] = adults.R

                if children.shape[1] == 4:
                    state_children_add[i, 2, :] = (children.R + children.D)
                else:
                    state_children_add[i, 2, :] = children.R

                J_vec_add[i] = J

        # Concatenate samples
        state_adults = np.concatenate((state_adults, state_adults_add), axis=0)
        state_children = np.concatenate((state_children, state_children_add), axis=0)
        J_vec = np.concatenate((J_vec, J_vec_add))

        # Update number of samples
        num_MC_sim = J_vec.shape[0]

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

# Mean
state_adults_mean = np.mean(state_adults, axis=0)
state_children_mean = np.mean(state_children, axis=0)

# Sample variance
state_adults_var = np.var(state_adults, axis=0, ddof=1)
state_children_var = np.var(state_children, axis=0, ddof=1)

# Sample standard deviation
state_adults_std = np.std(state_adults, axis=0, ddof=1)
state_children_std = np.std(state_children, axis=0, ddof=1)

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
