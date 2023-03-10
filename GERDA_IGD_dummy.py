#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:49:12 2022

Runs an inexact gradient descent algorithm to find the optimal control for the
dummy ABM.

@author: Jan-Hendrik Niemann
"""


import numpy as np
import os
import sys
import json
import time
from datetime import datetime
import math


def objective_estimate(u, settings_dict):
    """
    Estimate objective value of dummy ABM

    Parameters
    ----------
    u : ndarray
        Control vector.
    settings_dict : dict
        Dictionary with optimization settings.

    Returns
    -------
    J : float
        Estimate of objective function.
    e : float
        Error of objective estimate with e <= 2 * standard deviation of sample mean
    num_MC_sim_obj_est : int
        Number of Monte Carlo simulations needed to achive error e

    """

    working_directory = settings_dict['working_directory']

    # Export control u
    np.savetxt(os.path.join(data_path, 'control_U.csv'), u, delimiter=',')

    # Estimate J
    os.chdir(working_directory)
    os.system('chmod +x GERDA_state_estimation_dummy.py; ./GERDA_state_estimation_dummy.py');

    # Read sample mean of J from file
    J = np.genfromtxt(os.path.join(data_path, 'objective_estimate.csv'), delimiter=',')
    num_MC_sim_obj_est = np.genfromtxt(os.path.join(data_path, 'num_MC_sim_obj_est.csv'), delimiter=',', dtype=int).item()

    e = 2 * np.genfromtxt(os.path.join(data_path, 'objective_sample_mean_std.csv'), delimiter=',')

    return J, e, num_MC_sim_obj_est


def gradient_estimate(u, settings_dict):
    """
    Estimate objective gradient of dummy ABM

    Parameters
    ----------
    u : ndarray
        Control vector.
    settings_dict : dict
        Dictionary with optimization settings.

    Returns
    -------
    Ju : ndarray
        Gradient estimate.
    num_MC_sim : int
        Number of Monte Carlo simulations needed to achieve eps accuracy.

    """

    working_directory = settings_dict['working_directory']

    # Export control u
    np.savetxt(os.path.join(data_path, 'control_U.csv'), u, delimiter=',')

    # Estimate gradient
    os.chdir(working_directory)
    os.system('chmod +x GERDA_gradient_estimation_dummy.py; ./GERDA_gradient_estimation_dummy.py');

    # Read Ju, num_MC_sim from file and minimum relative accuracy gradient estimate
    Ju = np.genfromtxt(os.path.join(data_path, 'gradient_estimate_projected.csv'), delimiter=',')
    num_MC_sim = np.genfromtxt(os.path.join(data_path, 'num_MC_sim_grad_est.csv'), delimiter=',', dtype=int).item()
    eps = np.genfromtxt(os.path.join(data_path, 'eps.csv'), delimiter=',')

    return Ju, num_MC_sim, eps


# %% Parameters

# Path to settings file
path = '<path_to_input_and_output_directory>/opt_settings.txt'

with open(path, 'r') as file:
    settings_dict = json.load(file)

# Load settings
total_timesteps = settings_dict['total_timesteps']
duration_of_constant_control = settings_dict['duration_of_constant_control']
data_path = settings_dict['data_path']
output_dir = settings_dict['output_directory']
itr_max = settings_dict['max_iterations']
c1 = settings_dict['c1']
u_school_min = settings_dict['u_school_min']
u_school_max = settings_dict['u_school_max']
u_work_min = settings_dict['u_work_min']
u_work_max = settings_dict['u_work_max']

# Initial gradient descent step size
step_size_0 = 0.01

# Number of Monte Carlo simulations
num_MC_sim_obj_est = np.genfromtxt(os.path.join(data_path, 'num_MC_sim_obj_est.csv'), delimiter=',', dtype=int).item()
num_MC_sim_grad_est = np.genfromtxt(os.path.join(data_path, 'num_MC_sim_grad_est.csv'), delimiter=',', dtype=int).item()

# Set up controls
upper_bound_week = math.ceil(total_timesteps / duration_of_constant_control)
u_school = 0.0 * np.ones(upper_bound_week)
u_work = 0.0 * np.ones(upper_bound_week)
u0 = np.vstack((u_school, u_work)).T
u_min = np.empty_like(u0)
u_max = np.empty_like(u0)
u_min[:, 0] = u_school_min  # Lower bounds on control u
u_min[:, 1] = u_work_min  # Lower bounds on control u
u_max[:, 0] = u_school_max  # Upper bounds on control u
u_max[:, 1] = u_work_max  # Upper bounds on control u

# Export control u
np.savetxt(os.path.join(data_path, 'control_U.csv'), u0, delimiter=',')

# %% Run optimization

# Print settings into README file
with open(os.path.join(data_path, output_dir, 'README.txt'), 'w') as file:
    file.write('Inexact Gradient Descent Optimization dummy ABM\n\n')

    # List of keys
    lst = list(settings_dict)

    for i in range(len(lst)):
        file.write('%s: %s\n' % (lst[i], settings_dict[lst[i]]))

# Reset number of Monte Carlo samples
with open(os.path.join(data_path, 'num_MC_sim_grad_est.csv'), 'w') as file:
    file.write(str(settings_dict['grad_est_MC_sim_init']))

# Reset number of Monte Carlo samples
with open(os.path.join(data_path, 'num_MC_sim_obj_est.csv'), 'w') as file:
    file.write(str(settings_dict['obj_est_MC_sim_init']))

# Control vector
if settings_dict['resume_iterations_at'] > 0:
    start_itr_at = settings_dict['resume_iterations_at']
    print('Resume iterations at %g' % settings_dict['resume_iterations_at'])
    u = np.genfromtxt(os.path.join(data_path, output_dir, 'u_coarse_' + str(settings_dict['resume_iterations_at']) + '.csv'), delimiter=',')

    # Load vectors
    J_vec = np.genfromtxt(os.path.join(data_path, output_dir, 'J_vec.csv'), delimiter=',')
    ABM_evaluations_vec = np.genfromtxt(os.path.join(data_path, output_dir, 'ABM_evaluations_vec.csv'), delimiter=',')
else:
    u = u0
    np.savetxt(os.path.join(data_path, output_dir, 'u_coarse_0.csv'), u0, delimiter=',')
    start_itr_at = 0

    # Allocate vectors
    J_vec = np.zeros(itr_max + 1)
    ABM_evaluations_vec = np.zeros(itr_max)

# Initial step size
step_size = step_size_0

# Save gradient descent step size
with open(os.path.join(data_path, 'alpha.csv'), 'w') as file:
    file.write(str(step_size))

# Save finite difference step size
with open(os.path.join(data_path, 'rho.csv'), 'w') as file:
    file.write(str(settings_dict['fd_step_size']))

# Remove file with error_e for objective estimate as it is not needed for IGD
try:
    os.remove(os.path.join(data_path, 'error_e.csv'))
    print('Removed file "%s"' % os.path.join(data_path, 'error_e.csv'))
except OSError:
    print('File "%s" does not exist' % os.path.join(data_path, 'error_e.csv'))

# Set error
e = np.inf

tic = time.time()
for itr in range(start_itr_at, itr_max):
    counter_filename = 0

    now = datetime.now()
    dt_string = now.strftime('%H:%M:%S')

    print('\n+ + + Iteration No. %g of %g at %s + + +\n' % (itr + 1, itr_max, dt_string))

    # Estimate gradient
    print('Gradient estimate...')
    Ju, num_MC_sim_grad_est, eps = gradient_estimate(u, settings_dict)
    ABM_evaluations_vec[itr] += 4 * upper_bound_week * num_MC_sim_grad_est

    if np.isnan(eps) or np.isinf(eps) or np.isinf(np.linalg.norm(Ju, ord=2)):
        print('\nError while computing gradient estimates\n')
        sys.exit()

    # Descent direction
    du = -Ju

    # Choose step size
    a = step_size
    while a <= step_size_0:
        a = 1.1 * a
        if np.any(u + a * du > u_max) or np.any(u + a * du < u_min):
            # Revert last change
            a = a / 1.1
            break
    step_size = min(step_size_0, a)

    if itr == 0:
        # Save step size
        with open(os.path.join(data_path, 'alpha.csv'), 'w') as file:
            file.write(str(step_size))

        J, e, num_MC_sim_obj_est = objective_estimate(u, settings_dict)
        ABM_evaluations_vec[itr] += num_MC_sim_obj_est

        # Save objective
        J_vec[itr] = J

    while True:

        # Save step size
        with open(os.path.join(data_path, 'alpha.csv'), 'w') as file:
            file.write(str(step_size))

        print('Testing descent step size %g' % (step_size))

        # Estimate objective function
        if num_MC_sim_obj_est >= settings_dict['samplesize_max']:
            print('Objective J_old estimate...\nMaximum number of Monte Carlo simulations reached')
        elif e > eps * c1 * step_size * np.linalg.norm(Ju, ord=2)**2:
            print('Objective J_old estimate...')
            J, e, num_MC_sim_obj_est = objective_estimate(u, settings_dict)
            ABM_evaluations_vec[itr] += num_MC_sim_obj_est

        # Find new control
        u_new = np.minimum(np.maximum(u_min, u + step_size * du), u_max)

        # Estimate objective function
        print('Objective J_new estimate...')
        J_new, e, num_MC_sim_obj_est = objective_estimate(u_new, settings_dict)
        ABM_evaluations_vec[itr] += num_MC_sim_obj_est

        print('J_old = %g' % J)
        print('J_new = %g' % J_new)
        print('|Ju| = %g' % np.linalg.norm(Ju, ord=2))

        # Acceptance test based on Armijo rule
        if J_new - J <= -(1 + 3 * eps) * c1 * step_size * (np.linalg.norm(du, ord=2))**2:
            print('Acceptance test passed with J_new = %g < J_old = %g' % (J_new, J))
            u = u_new
            J = J_new

            break
        elif step_size < 1e-6:
            step_size = step_size_0
            print('Gradient descent step size too small...\nReset step size to %g' % step_size)

            break
        else:
            print('Reject step...\n')
            step_size = step_size / 3

            # Save rejected controls
            np.savetxt(os.path.join(data_path, output_dir, 'u_coarse_' + str(itr + 1) + '_' + str(counter_filename) + '.csv'), u_new, delimiter=',')
            counter_filename += 1

    # Save objective
    J_vec[itr + 1] = J

    # Save to csv files
    np.savetxt(os.path.join(data_path, output_dir, 'J_vec.csv'), J_vec, delimiter=',')
    np.savetxt(os.path.join(data_path, output_dir, 'Ju_' + str(itr) + '.csv'), Ju, delimiter=',')
    np.savetxt(os.path.join(data_path, output_dir, 'u_coarse_' + str(itr + 1) + '.csv'), u, delimiter=',')
    np.savetxt(os.path.join(data_path, output_dir, 'ABM_evaluations_vec.csv'), ABM_evaluations_vec, delimiter=',')

toc = time.time()
print('\nElapsed time %g minutes' % ((toc - tic) / 60))
