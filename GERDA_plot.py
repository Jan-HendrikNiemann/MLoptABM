#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:51:15 2022

Demonstration of how the output data is plotted

@author: Jan-Hendrik Niemann
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json
import os
import math
import copy

from GERDA_aux import epidemic_ODE, load, baseline_scenario


# %% Load and create data

# Path to settings file
path = '<path_to_input_and_output_directory>/opt_settings.txt'

with open(path, 'r') as file:
    settings_dict = json.load(file)

total_timesteps = settings_dict['total_timesteps']
duration_of_constant_control = settings_dict['duration_of_constant_control']
n_agents = settings_dict['n_agents']
data_path = settings_dict['data_path']
output_dir = settings_dict['output_directory']

# Load data
adults_mean = np.genfromtxt(os.path.join(data_path, output_dir, 'adults_state_estimate_mean.csv'), delimiter=',')
children_mean = np.genfromtxt(os.path.join(data_path, output_dir, 'children_state_estimate_mean.csv'), delimiter=',')
adults_std = np.genfromtxt(os.path.join(data_path, output_dir, 'adults_state_estimate_std.csv'), delimiter=',')
children_std = np.genfromtxt(os.path.join(data_path, output_dir, 'children_state_estimate_std.csv'), delimiter=',')

# Tube around mean
adults_upper = (adults_mean + adults_std)/n_agents
adults_lower = (adults_mean - adults_std)/n_agents
children_upper = (children_mean + children_std)/n_agents
children_lower = (children_mean - children_std)/n_agents

# Integrate ODE with fitting parameters
time = np.linspace(0, total_timesteps/24, total_timesteps)
y0 = np.genfromtxt(os.path.join(data_path, output_dir, 'y0.csv'), delimiter=',')
p = np.genfromtxt(os.path.join(data_path, output_dir, 'fitting_parameter.csv'), delimiter=',')
sol = solve_ivp(epidemic_ODE, [0, total_timesteps/24], y0, args=(p, np.zeros(2)), max_step=1/24)

# %% Mean, standard deviation and fitted ODE

print('Creating plot: mean, standard deviation and fitted ODE...')

fig = plt.figure()
plt.plot(sol.t, sol.y.T/n_agents, 'k')
plt.plot(time, adults_mean[:, 0]/n_agents, 'b', label=r'$S_\mathrm{a}$')
plt.fill_between(time, adults_upper[:, 0], adults_lower[:, 0], color='b', alpha=0.2)
plt.plot(time, adults_mean[:, 1]/n_agents, 'r', label=r'$I_\mathrm{a}$')
plt.fill_between(time, adults_upper[:, 1], adults_lower[:, 1], color='r', alpha=0.2)
plt.plot(time, adults_mean[:, 2]/n_agents, 'g', label=r'$R_\mathrm{a}$')
plt.fill_between(time, adults_upper[:, 2], adults_lower[:, 2], color='g', alpha=0.2)
plt.plot(time, children_mean[:, 0]/n_agents, 'b--', label=r'$S_\mathrm{c}$')
plt.fill_between(time, children_upper[:, 0], children_lower[:, 0], color='b', alpha=0.2)
plt.plot(time, children_mean[:, 1]/n_agents, 'r--', label=r'$I_\mathrm{c}$')
plt.fill_between(time, children_upper[:, 1], children_lower[:, 1], color='r', alpha=0.2)
plt.plot(time, children_mean[:, 2]/n_agents, 'g--', label=r'$R_\mathrm{c}$')
plt.fill_between(time, children_upper[:, 2], children_lower[:, 2], color='g', alpha=0.2)
plt.legend(ncol=2)
plt.xlabel('Time [days]')
plt.ylabel('Fraction of agents')
plt.savefig(os.path.join(data_path, output_dir, 'ABM_ODE_fit'), bbox_inches='tight', transparent=True)

# %% Run and plot ABM without controls

print('Running and creating plot: ABM without controls...')

# Load world
modeledWorld = load(settings_dict, False)

upper_bound_week = math.ceil(total_timesteps / duration_of_constant_control)

Us = np.zeros(upper_bound_week)
Uw = np.zeros(upper_bound_week)

adults, children = baseline_scenario(copy.deepcopy(modeledWorld), Us, Uw, settings_dict)

fig = plt.figure()
plt.step(time, adults.S/n_agents, 'b', label=r'$S_\mathrm{a}$')
plt.step(time, adults.I/n_agents, 'r', label=r'$I_\mathrm{a}$')
plt.step(time, adults.R/n_agents, 'g', label=r'$R_\mathrm{a}$')
plt.step(time, children.S/n_agents, 'b--', label=r'$S_\mathrm{c}$')
plt.step(time, children.I/n_agents, 'r--', label=r'$I_\mathrm{c}$')
plt.step(time, children.R/n_agents, 'g--', label=r'$R_\mathrm{c}$')
plt.legend(ncol=2)
plt.xlabel('Time [days]')
plt.ylabel('Fraction of agents')
plt.savefig(os.path.join(data_path, output_dir, 'ABM_no_ctrl'), bbox_inches='tight', transparent=True)

# %% Optimal control

print('Creating plot: optimal controls...')

U = np.genfromtxt(os.path.join(data_path, output_dir, 'u_coarse_' + str(settings_dict['max_iterations'] - 1) + '.csv'), delimiter=',')

if U.ndim == 1:
    U = np.expand_dims(U, 0)

fig = plt.figure()
plt.step(np.linspace(0, total_timesteps/24, U.shape[0] + 1), np.vstack((U, U[-1, :])), where='post')
plt.xlabel('Time [days]')
plt.ylabel('Control $u(t)$')
plt.legend(('$u_\mathrm{s}(t)$', '$u_\mathrm{w}(t)$'))
plt.ylim((-0.05, 1.05))
plt.savefig(os.path.join(data_path, output_dir, 'opt_ctrl'), bbox_inches='tight', transparent=True)

# %% Run and plot ABM with optimal controls

print('Running and creating plot: ABM with optimal controls...')

duration_of_constant_control_in_days = int(duration_of_constant_control / 24)

Us = U[:, 0]
Uw = U[:, 1]

adults, children = baseline_scenario(copy.deepcopy(modeledWorld), Us, Uw, settings_dict)

time = np.linspace(0, total_timesteps/24, total_timesteps)

fig = plt.figure()
plt.step(time, adults.S/n_agents, 'b', label=r'$S_\mathrm{a}$')
plt.step(time, adults.I/n_agents, 'r', label=r'$I_\mathrm{a}$')
plt.step(time, adults.R/n_agents, 'g', label=r'$R_\mathrm{a}$')
plt.step(time, children.S/n_agents, 'b--', label=r'$S_\mathrm{c}$')
plt.step(time, children.I/n_agents, 'r--', label=r'$I_\mathrm{c}$')
plt.step(time, children.R/n_agents, 'g--', label=r'$R_\mathrm{c}$')
plt.legend(ncol=2)
plt.xlabel('Time [days]')
plt.ylabel('Fraction of agents')
plt.savefig(os.path.join(data_path, output_dir, 'ABM_opt_ctrl'), bbox_inches='tight', transparent=True)

# %% Mean and standard deviation with optimal controls

print('Creating plot: mean, standard deviation with optimal control...')

# Load data
adults_mean = np.genfromtxt(os.path.join(data_path, output_dir, 'adults_state_estimate_mean_tmp.csv'), delimiter=',')
children_mean = np.genfromtxt(os.path.join(data_path, output_dir, 'children_state_estimate_mean_tmp.csv'), delimiter=',')
adults_std = np.genfromtxt(os.path.join(data_path, output_dir, 'adults_state_estimate_std_tmp.csv'), delimiter=',')
children_std = np.genfromtxt(os.path.join(data_path, output_dir, 'children_state_estimate_std_tmp.csv'), delimiter=',')

# Tube around mean
adults_upper = (adults_mean + adults_std)/n_agents
adults_lower = (adults_mean - adults_std)/n_agents
children_upper = (children_mean + children_std)/n_agents
children_lower = (children_mean - children_std)/n_agents

fig = plt.figure()
plt.plot(time, adults_mean[:, 0]/n_agents, 'b', label=r'$S_\mathrm{a}$')
plt.fill_between(time, adults_upper[:, 0], adults_lower[:, 0], color='b', alpha=0.2)
plt.plot(time, adults_mean[:, 1]/n_agents, 'r', label=r'$I_\mathrm{a}$')
plt.fill_between(time, adults_upper[:, 1], adults_lower[:, 1], color='r', alpha=0.2)
plt.plot(time, adults_mean[:, 2]/n_agents, 'g', label=r'$R_\mathrm{a}$')
plt.fill_between(time, adults_upper[:, 2], adults_lower[:, 2], color='g', alpha=0.2)
plt.plot(time, children_mean[:, 0]/n_agents, 'b--', label=r'$S_\mathrm{c}$')
plt.fill_between(time, children_upper[:, 0], children_lower[:, 0], color='b', alpha=0.2)
plt.plot(time, children_mean[:, 1]/n_agents, 'r--', label=r'$I_\mathrm{c}$')
plt.fill_between(time, children_upper[:, 1], children_lower[:, 1], color='r', alpha=0.2)
plt.plot(time, children_mean[:, 2]/n_agents, 'g--', label=r'$R_\mathrm{c}$')
plt.fill_between(time, children_upper[:, 2], children_lower[:, 2], color='g', alpha=0.2)
plt.legend(ncol=2)
plt.xlabel('Time [days]')
plt.ylabel('Fraction of agents')
plt.savefig(os.path.join(data_path, output_dir, 'ABM_opt_ctrl_mean_std'), bbox_inches='tight', transparent=True)

# %% Create error plots

print('Creating error plots...')

J_vec = np.genfromtxt(os.path.join(data_path, output_dir, 'J_vec.csv'), delimiter=',')
J_vec = J_vec[~np.isnan(J_vec)]

total_samples = np.genfromtxt(os.path.join(data_path, output_dir, 'ABM_evaluations_vec.csv'), delimiter=',')
total_samples = total_samples[np.nonzero(total_samples)]

fig = plt.figure()
plt.loglog(np.cumsum(total_samples)[:-1], np.abs(J_vec[:-1] - J_vec[-1]))
plt.xlabel('No. ABM simulations')
plt.ylabel('err')
plt.grid()

plt.savefig(os.path.join(data_path, output_dir, 'error_ABM_evaluations'), bbox_inches='tight', transparent=True)

fig = plt.figure()
plt.semilogy(np.arange(1, len(J_vec)), np.abs(J_vec[:-1] - J_vec[-1]))
plt.xlabel('Iteration $k$')
plt.ylabel('err')
plt.grid()
plt.savefig(os.path.join(data_path, output_dir, 'error_iterations'), bbox_inches='tight', transparent=True)
