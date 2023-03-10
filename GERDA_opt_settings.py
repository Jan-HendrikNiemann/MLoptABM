#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:09:22 2022

This script initializes all the necessary parameters. It also create necessary
initial files and a unique output directory to avoid loss of data if requested
by the user

@author: Jan-Hendrik Niemann
"""


import json
import os
import re
import numpy as np


# Settings
path_to_GERDA = '<path_to_working_GERDA_directory>'
working_directory = '<path_to_working_directory>'
data_path = '<path_to_working_directory>/data'
geopath = 'input_data/geo/'
geofiles = {0: 'Buildings_Gangelt_MA_1.csv',
            1: 'Buildings_Gangelt_MA_3.csv'}
world_to_pick = 1  # Version of modeled town (Gangelt)
n_initially_infected = 5  # Number of initially infected persons
total_timesteps = 49 * 24  # Time in [hours], days * 24
duration_of_constant_control = 7 * 24  # Time in [hours]
general_infectivity = 0.175  # GERDA parameter
general_interaction_frequency = 1  # GERDA parameter
fd_step_size = 0.1  # Initial finite difference step size
eps = 0.25  # Relative accuracy of gradient estimate
c1 = 0.1  # Fraction of descent promised
samplesize_max = 1e6  # Maximum number of objective and gradient estimates
saved_world = 'Reduced_Gangelt_n1096_worldObj.pkl'  # Name of GERDA world object
n_agents = 1096  # Number of agents of the GERDA model
output_directory = 'output_'
rho = 0.5  # Initial trust region radius
rho_min = 1e-4  # Minimal trust region radius
resume_iterations_at = 0
max_iterations = 15  # Number of iterations
max_coarse_steps = 100  # Number of coarse model iterations
obj_est_MC_sim_init = 100  # Initial number of objective estimates
grad_est_MC_sim_init = 100  # Initial number of gradient estimates
fit_ODE = False  # Fit ODE in every iteration
u_school_min = 0  # Bounds on controls
u_school_max = 1  # Bounds on controls
u_work_min = 0  # Bounds on controls
u_work_max = 0.8  # Bounds on controls
fast_mode = False  # Use ODE with noise instead of Markov jump process

# %%


def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


# Create dictionary with settings
x = {"path_to_GERDA": path_to_GERDA,
     "working_directory": working_directory,
     "geopath": geopath,
     "geofiles": geofiles,
     "world_to_pick": world_to_pick,
     "n_initially_infected": n_initially_infected,
     "total_timesteps": total_timesteps,
     "duration_of_constant_control": duration_of_constant_control,
     "general_infectivity": general_infectivity,
     "general_interaction_frequency": general_interaction_frequency,
     "fd_step_size": fd_step_size,
     "data_path": data_path,
     "eps": eps,
     "c1": c1,
     "samplesize_max": samplesize_max,
     "saved_world": saved_world,
     "n_agents": n_agents,
     "output_directory": output_directory,
     "rho": rho,
     "rho_min": rho_min,
     # "max_recursive_trust_region_steps": max_recursive_trust_region_steps,
     "max_coarse_steps": max_coarse_steps,
     "resume_iterations_at": resume_iterations_at,
     "max_iterations": max_iterations,
     "fit_ODE": fit_ODE,
     "u_school_min": u_school_min,
     "u_school_max": u_school_max,
     "u_work_min": u_work_min,
     "u_work_max": u_work_max,
     "obj_est_MC_sim_init": obj_est_MC_sim_init,
     "grad_est_MC_sim_init": grad_est_MC_sim_init,
     "fast_mode": fast_mode
     }

resume_work = False

# Ask if new output directory is requested by the user
new_out_dir = input('Do you want to create a new output directory? ')
new_out_dir = new_out_dir.lower()

while new_out_dir not in {'yes', 'no'}:
    new_out_dir = input('Do you want to create a new output directory? ')
    new_out_dir = new_out_dir.lower()

if new_out_dir in {'yes'}:
    new_output_dir = True
else:
    new_output_dir = False
    print('Current data directory is', data_path)
    old_out_dir = input('Please enter the name of an existing output directory: ')

    while not os.path.isdir(os.path.join(data_path, old_out_dir)):
        print('%s is not an existing directory' % old_out_dir)
        old_out_dir = input('Please enter the name of an existing output directory: ')

    x['output_directory'] = old_out_dir

    resume_work = input('Do you want to resume your work? ')
    resume_work = resume_work.lower()

    while new_out_dir not in {'yes', 'no'}:
        resume_work = input('Do you want to resume your work? ')
        resume_work = resume_work.lower()

    if resume_work in {'yes'}:
        resume_work = True
        list_of_files = []

        print(r'These are all files Ju_*.csv I have found:')
        for file in get_files(os.path.join(data_path, old_out_dir)):
            if 'Ju_' in file:
                list_of_files.append(file)
                print(file)

        itr = 0
        for file in list_of_files:
            num = int(re.search('Ju_(\d*)', file).group(1))  # Assuming filename is "Ju_xxx.csv"
            if num > itr:
                itr = num
            else:
                itr
        print('The last iteration was %g' % itr)

        resume_work_at = input('Do you want to resume your work with iteration %g? ' % (itr + 1))
        resume_work_at = resume_work_at.lower()

        while new_out_dir not in {'yes', 'no'}:
            resume_work_at = input('Do you want to resume your work with iteration %g? ' % (itr + 1))
            resume_work_at = resume_work_at.lower()

        if resume_work_at in {'yes'}:
            x['resume_iterations_at'] = itr + 1
        else:
            itr = input('At which step do you want to resume your work?')

            while not isinstance(itr, int):
                itr = input('Please enter an interger!')

            x['resume_iterations_at'] = itr

    else:
        resume_work = False
        print('Start iteration at %g' % resume_iterations_at)


# Create unique output directory if requested by the user
counter = 0
while new_output_dir:
    try:
        os.mkdir(os.path.join(data_path, output_directory + str(counter)))
        print('New output directory created at %s' % os.path.join(data_path, output_directory + str(counter)))

        # Update output directory
        x['output_directory'] = output_directory + str(counter)

        break
    except OSError:
        counter = counter + 1

# Save settings as json encoded file
with open(os.path.join(data_path, 'opt_settings.txt'), 'w') as json_file:
    json.dump(x, json_file, indent=4, sort_keys=True, ensure_ascii=True)

# Initialize control
if resume_work:
    control_U = np.genfromtxt(os.path.join(data_path, x['output_directory'], 'u_coarse_' + str(itr) + '.csv'), delimiter=',')
    np.savetxt(os.path.join(data_path, 'control_U.csv'), control_U, delimiter=',')
else:
    np.savetxt(os.path.join(data_path, 'control_U.csv'), np.zeros((int(total_timesteps / duration_of_constant_control), 2)), delimiter=',')
    print('Created control_U.csv with zeros only')

    # Initialize Monte Carlo simulation number for J estimates
    with open(os.path.join(data_path, 'num_MC_sim_obj_est.csv'), 'w') as file:
        file.write(str(obj_est_MC_sim_init))
    print('Created num_MC_sim_obj_est.csv with initially %i Monte Carlo simulations' % obj_est_MC_sim_init)

    # Initialize Monte Carlo simulation number for gradient Ju estimates
    with open(os.path.join(data_path, 'num_MC_sim_grad_est.csv'), 'w') as file:
        file.write(str(grad_est_MC_sim_init))
    print('Created num_MC_sim_grad_est.csv with initially %i Monte Carlo simulations' % grad_est_MC_sim_init)
