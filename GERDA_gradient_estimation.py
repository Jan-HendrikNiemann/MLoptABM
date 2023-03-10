#!<path_to_GERDA_virtuel_environment>/envs/gerdaenv/bin/python3.8
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:52:19 2022

This script computes in parallel Monte Carlo estimates of the gradient of the
objective function with respect to the control using finite differences. The
inputs are automatically loaded from the data_path specified in
opt_settings.txt. The results are saved in the same directory.

@author: Jan-Hendrik Niemann
"""


import numpy as np
import math
import os
import json
import time
from joblib import Parallel, delayed
import multiprocessing

from GERDA_aux import load, gradient, condense_vec, gradient_projection


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
eps_min = settings_dict['eps']
samplesize_max = settings_dict['samplesize_max']

# Load control vector
U = np.genfromtxt(os.path.join(data_path, 'control_U.csv'), delimiter=',')
if U.ndim == 1:
    U = np.expand_dims(U, 0)
Us = U[:, 0]
Uw = U[:, 1]

# Number of available cores for parallelization
num_cores = multiprocessing.cpu_count()

# Number of Monte Carlo simulations
samplesize = np.genfromtxt(os.path.join(data_path, 'num_MC_sim_grad_est.csv'), delimiter=',', dtype=int).item()

if samplesize <= num_cores:
    num_cores = samplesize

# %% Execute

# Create empty array
Ju_samples = np.zeros((0, 2, total_timesteps))

# Initialize relative error
eps = np.inf

rho = np.genfromtxt(os.path.join(settings_dict['data_path'], 'rho.csv'), delimiter=',')
fd_step_size = min(10 ** math.floor(math.log10(rho)), settings_dict['fd_step_size'])
print('Finite differences step size h = %g' % fd_step_size)

with Parallel(n_jobs=num_cores, verbose=0) as parallel:
    while eps >= eps_min:

        samplesize_add = int(samplesize * 0.5)

        # Random variable for better seed
        rnd_var = np.random.randint(0, 2**32 - 1)

        # If no samples are done yet, create initial samples
        if Ju_samples.shape[0] == 0:
            result = parallel(delayed(gradient)(modeledWorld, Us, Uw, settings_dict, sim_id + rnd_var) for sim_id in range(samplesize))
            Ju_samples_add = np.asarray(result, dtype=float)

        # If accuracy is too low, create more samples
        else:
            print('Accuracy eps = %g !< %g = eps_min' % (eps, eps_min))
            result = parallel(delayed(gradient)(modeledWorld, Us, Uw, settings_dict, sim_id + rnd_var) for sim_id in range(samplesize_add))
            Ju_samples_add = np.asarray(result, dtype=float)

        Ju_samples = np.concatenate((Ju_samples, Ju_samples_add), axis=0)

        Ju = np.mean(Ju_samples, axis=0)

        # Covariance & Std. deviation
        Ju_samples_unique = Ju_samples[:, :, ::duration_of_constant_control]

        # -1 to get a 2d array with samplesize rows and "suitable" columns
        Ju_samples_unique = Ju_samples_unique.reshape((Ju_samples_unique.shape[0], -1), order='F')

        # Reduce size
        Ju = condense_vec(Ju, duration_of_constant_control)
        Ju_flat = Ju.flatten(order='F')

        covar_matrix = np.cov(Ju_samples_unique, rowvar=False)
        std = np.sqrt(np.linalg.norm(covar_matrix, ord=2) / Ju_samples_unique.shape[0])
        eps = 2 * std / np.linalg.norm(Ju_flat, ord=2)

        # If maximum sample size is reach, stop increasing
        if Ju_samples_unique.shape[0] >= samplesize_max:
            print('Maximum number of Monte Carlo simulations reached')
            break

samplesize = Ju_samples_unique.shape[0]

# Project gradient
Ju_projected = -gradient_projection(-Ju.T, U, settings_dict).T

print('Relative error eps = %g with %g Monte Carlo simulations' % (eps, samplesize))
print('Sample mean standard deviation std = %g' % std)
print('|Ju| = %g' % np.linalg.norm(Ju_flat, ord=2))
print('|V| = %g' % np.linalg.norm(covar_matrix, ord=2))

print('Elapsed time %g minutes\n' % ((time.time() - tic) / 60))

# %% Save to csv files

np.savetxt(os.path.join(data_path, 'gradient_estimate.csv'), Ju.T, delimiter=',')
np.savetxt(os.path.join(data_path, 'gradient_estimate_projected.csv'), Ju_projected.T, delimiter=',')

with open(os.path.join(data_path, 'num_MC_sim_grad_est.csv'), 'w') as file:
    file.write(str(samplesize))

with open(os.path.join(data_path, 'eps.csv'), 'w') as file:
    file.write(str(eps))

# %% Save auxiliray output and samples

# Create dictionary with settings
out_dict = {"eps": eps,
            "samplesize": samplesize,
            "sample_mean_standard_deviation": std
            }

counter = 0
while os.path.exists(os.path.join(data_path, output_dir, 'Ju_aux_out_' + str(counter) + '.json')):
    counter += 1

# Save as json encoded file
with open(os.path.join(data_path, output_dir, 'Ju_aux_out_' + str(counter) + '.json'), 'w') as json_file:
    json.dump(out_dict, json_file, indent=4, sort_keys=True, ensure_ascii=True)

np.savez_compressed(os.path.join(data_path, output_dir, 'Ju_samples_' + str(counter) + '.npz'), Ju_samples=Ju_samples_unique)
