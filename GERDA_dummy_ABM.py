#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:28:35 2020

SIR Markov model fitted to GERDA. For infinitly many agent this model behaves
like the coarse ODE approximation

@author: Jan-Hendrik Niemann
"""

import numpy as np
import math
import os
import random
from copy import deepcopy

from GERDA_aux import objective, stretch_vec, condense_vec


def epidemic_ODE(t, y, p, u):
    """
    Right-hand side of SIRS ODE model + Gaussian noise

    Parameters
    ----------
    t : ndarray
        Time.
    y : ndarray
        State vector.
    p : ndarray
        Parameter vector.
    u : ndarray
        Control vector u = [u_school, u_homeoffice].

    Returns
    -------
    f : ndarray
        Left-hand side ODE.

    """

    # Susceptible adults/children
    Sa = y[0]
    Sc = y[1]

    # Infected adults/children
    Ia = y[2]
    Ic = y[3]

    # Policies
    P1 = p[0] * (1 - u[1])**2
    P2 = p[1] * (1 - 0.5 * u[0]) * (1 - 0.5 * u[1])
    P3 = p[2] * (1 - u[0])**2

    # Transition rates between compartments + Gaussian noise
    Sa_Ia = Sa * (P1 * Ia + P2 * Ic) + random.gauss(0, 1)
    Sc_Ic = Sc * (P3 * Ic + P2 * Ia) + random.gauss(0, 1)
    Ia_Ra = p[3] * Ia + random.gauss(0, 1)
    Ic_Rc = p[4] * Ic + random.gauss(0, 1)

    f = np.array([-Sa_Ia + 0.2 * Ia_Ra,
                  -Sc_Ic + 0.2 * Ic_Rc,
                  Sa_Ia - Ia_Ra,
                  Sc_Ic - Ic_Rc,
                  0.8 * Ia_Ra,
                  0.8 * Ic_Rc])

    return f


def control(t, u, settings_dict):
    """
    Auxiliary function to obtain the control at time t

    Parameters
    ----------
    t : float
        Point of time (continuous).
    u : ndarray
        Condensed control vector.
    settings_dict : dict
        Dictionary with optimization settings.

    Returns
    -------
    U : ndarray
        Control at time t.

    """
    duration_of_constant_control = settings_dict['duration_of_constant_control']
    t_idx = int(math.floor(t / (duration_of_constant_control / 24)))  # 24 hours

    if u.ndim == 1:
        u = np.expand_dims(u, 0)
    U = u[t_idx, :]

    return U


def ODE_with_noise(x_init, p, u, t_step, T_max, settings_dict, seed=None):
    """
    Integrate epidemic ODE. This function has the same arguments as the
    function markov_jump_process(...) to check gradient computation for
    correctness.

    Parameters
    ----------
    x_init : ndarray
        Initial state.
    p : ndarray
        Fitting parameter.
    u : ndarray
        Condensed control vector.
    t_step : float
        Time step for output.
    T_max : int or float
        Time horizon.
    settings_dict : dict
        Dictionary with optimization settings.
    seed : int, optional
        Seed of random process. The default is None.

    Returns
    -------
    X : ndarray
        Trajectory of the system state for given time horizon.

    """

    h = 1 / 24

    u = stretch_vec(u[:, 0], u[:, 1], settings_dict['total_timesteps'], settings_dict['duration_of_constant_control'])

    N = u.shape[1]

    # Allocate memory
    Y = np.zeros((len(x_init), N))

    # Sort input
    Y[0, 0] = x_init[0]
    Y[1, 0] = x_init[3]
    Y[2, 0] = x_init[1]
    Y[3, 0] = x_init[4]
    Y[4, 0] = x_init[2]
    Y[5, 0] = x_init[5]

    # 4th order Runge-Kutta scheme
    for i in range(N - 1):

        k1 = epidemic_ODE(None, Y[:, i], p, u[:, i])
        k2 = epidemic_ODE(None, Y[:, i] + h * k1/2, p, u[:, i])
        k3 = epidemic_ODE(None, Y[:, i] + h * k2/2, p, u[:, i])
        k4 = epidemic_ODE(None, Y[:, i] + h * k3, p, u[:, i])

        Y[:, i + 1] = Y[:, i] + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Sort output
    Y_sorted = np.zeros_like(Y)
    Y_sorted[0, :] = Y[0, :]  # Adults susceptible
    Y_sorted[1, :] = Y[2, :]  # Adults infected
    Y_sorted[2, :] = Y[4, :]  # Adults recovered
    Y_sorted[3, :] = Y[1, :]  # Children susceptible
    Y_sorted[4, :] = Y[3, :]  # Children infected
    Y_sorted[5, :] = Y[5, :]  # Children recovered

    return Y_sorted


def markov_jump_process(x_init, p, u, t_step, T_max, settings_dict, seed=None):
    """
    SIR model fitted to GERDA implemented as Markov jump process. For infinitly
    many agent this model behaves like the coarse ODE approximation

    Parameters
    ----------
    x_init : ndarray
        Initial state.
    p : ndarray
        Fitting parameter.
    u : ndarray
        Condensed control vector.
    t_step : float
        Time step for output.
    T_max : int or float
        Time horizon.
    settings_dict : dict
        Dictionary with optimization settings.
    seed : int, optional
        Seed of random process. The default is None.

    Returns
    -------
    X : ndarray
        Trajectory of the system state for given time horizon.

    """

    if settings_dict['fast_mode']:
        X = ODE_with_noise(x_init, p, u, t_step, T_max, settings_dict, seed)
        return X

    # Set random seed
    if seed is not None:
        random.seed(seed)

    # Number of agents
    num_agents = sum(x_init)

    # Number of timesteps for saving
    num_timesteps = int(np.round(T_max / t_step)) + 1

    # Number of types
    num_types = x_init.shape[0]

    # Allocate memory
    X = np.zeros((num_types, num_timesteps), dtype=np.float64)
    alpha = np.zeros((num_types, num_types), dtype=np.float64)

    # State
    x = deepcopy(x_init)
    x_out = [deepcopy(x_init)]

    # Time
    t = [0.0]

    # Transition rate constants
    gamma = np.zeros((num_types, num_types))
    gamma_prime = np.zeros_like(gamma)
    gamma_prime[1, 2] = p[3]  # Get recovered, Ia -> Ra
    gamma_prime[2, 0] = 0.2 * p[3]  # Get susceptible again, Ra -> Sa
    gamma_prime[4, 5] = p[4]  # Get recovered, Ic -> Rc
    gamma_prime[5, 3] = 0.2 * p[4]  # Get susceptible again, Rc -> Sc

    # Transition rate constants Sa -> Ia, Sc -> Ic due to Ic and Ia, resp.
    gamma_prime_prime = np.zeros_like(gamma)

    k = 0
    while t[k] < T_max:

        # Get control at time point t[k]
        U = control(t[k], u, settings_dict)

        # Transition rate constants, parameter p corrected by fitted number of agents
        gamma[0, 1] = settings_dict['n_agents'] * p[0] * (1 - U[1])**2  # Get infected, Sa -> Ia
        gamma[3, 4] = settings_dict['n_agents'] * p[2] * (1 - U[0])**2  # Get infected, Sc -> Ic

        # Transition rate constants, parameter p corrected by fitted number of agents
        gamma_prime_prime[0, 1] = settings_dict['n_agents'] * p[1] * (1 - 0.5 * U[0]) * (1 - 0.5 * U[1])
        gamma_prime_prime[3, 4] = settings_dict['n_agents'] * p[1] * (1 - 0.5 * U[0]) * (1 - 0.5 * U[1])

        for i in range(num_types):
            for j in range(num_types):
                alpha[i, j] = (gamma[i, j] / num_agents * x[i] * x[j]
                               + gamma_prime[i, j] * x[i]
                               + gamma_prime_prime[i, j] / num_agents * x[i] * x[-1-j])
            sum_alpha = np.sum(alpha, axis=1)

        lmbda = np.sum(alpha)

        if lmbda == 0:
            break

        p_rnd = random.uniform(0, 1)

        # Determine time for the event to happen
        tau = 1 / lmbda * math.log(1 / p_rnd)

        # Save time point
        t.append(t[k] + tau)

        p_rnd = random.uniform(0, 1)
        i = 0
        while sum(sum_alpha[:i + 1]) / lmbda < p_rnd:
            i = i + 1

        # Determine event to happen at time
        j = 0
        if i == 0:
            while sum(alpha[i, :j + 1]) / lmbda < p_rnd:
                j = j + 1
        else:
            while sum(sum_alpha[:i]) / lmbda + sum(alpha[i, :j + 1]) / lmbda < p_rnd:
                j = j + 1

        k = k + 1
        x[i] = x[i] - 1
        x[j] = x[j] + 1

        x_out.append(deepcopy(x))

    # Save for output
    t = np.asarray(t)
    x = np.asarray(x_out).T

    for i in range(num_timesteps):
        idx = np.argmin(t <= i * t_step) - 1
        X[:, i] = x[:, idx]

    return X


def gradient(x_init, p, u, t_step, T_max, settings_dict, sim_ID):
    """
    Compute the gradient of the objective function at control u via finite
    differences

    Parameters
    ----------
    x_init : ndarray
        Initial state.
    p : ndarray
        Fitting parameter.
    u : ndarray
        Condensed control vector.
    t_step : float
        Time step for output.
    T_max : int or float
        Time horizon.
    settings_dict : dict
        Dictionary with optimization settings.
    sim_ID : int
        Unique simulation identifier. Serves also as seed.

    Returns
    -------
    Ju : ndarray
        Gradient of the objective function at control u. For convenience this
        is a 2 x m matrix.

    """

    # Load settings
    total_timesteps = settings_dict['total_timesteps']
    duration_of_constant_control = settings_dict['duration_of_constant_control']
    u_school_min = settings_dict['u_school_min']
    u_school_max = settings_dict['u_school_max']
    u_work_min = settings_dict['u_work_min']
    u_work_max = settings_dict['u_work_max']

    # Adjust finite difference step size to trust region radius. They should
    # have the same order of magnitude
    rho = np.genfromtxt(os.path.join(settings_dict['data_path'], 'rho.csv'), delimiter=',')
    fd_step_size = min(10 ** math.floor(math.log10(rho)), settings_dict['fd_step_size'])

    # Allocate memory
    Ju = np.zeros((2, total_timesteps))

    Us = u[:, 0]
    Uw = u[:, 1]

    # Number of complete weeks
    k_max = u.shape[0]

    for j in range(Ju.shape[0]):
        for k in range(k_max - 1):

            V = np.zeros((2, u.shape[0]))
            V[j, k] = 0.5 * fd_step_size

            if np.any(u[:, 0] + V.T[:, 0] > u_school_max) or np.any(u[:, 1] + V.T[:, 1] > u_work_max):
                # Central
                SIR_all = markov_jump_process(x_init, p, u, t_step, T_max, settings_dict, sim_ID)
                adults = SIR_all[(0, 1, 2), :].T
                children = SIR_all[(3, 4, 5), :].T

                Jv = objective(adults, children,
                               stretch_vec(Us, Uw,
                                           total_timesteps, duration_of_constant_control), settings_dict)

                # Backward
                SIR_all = markov_jump_process(x_init, p, u - 2 * V.T, t_step, T_max, settings_dict, sim_ID)
                adults = SIR_all[(0, 1, 2), :].T
                children = SIR_all[(3, 4, 5), :].T

                Jw = objective(adults, children,
                               stretch_vec(Us - 2 * V[0, :], Uw - 2 * V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

            elif np.any(u[:, 0] + V.T[:, 0] > u_school_min) or np.any(u[:, 1] + V.T[:, 1] > u_work_min):
                # Forward
                SIR_all = markov_jump_process(x_init, p, u + 2 * V.T, t_step, T_max, settings_dict, sim_ID)
                adults = SIR_all[(0, 1, 2), :].T
                children = SIR_all[(3, 4, 5), :].T

                Jv = objective(adults, children,
                               stretch_vec(Us + 2 * V[0, :], Uw + 2 * V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

                # Central
                SIR_all = markov_jump_process(x_init, p, u, t_step, T_max, settings_dict, sim_ID)
                adults = SIR_all[(0, 1, 2), :].T
                children = SIR_all[(3, 4, 5), :].T

                Jw = objective(adults, children,
                               stretch_vec(Us, Uw,
                                           total_timesteps, duration_of_constant_control), settings_dict)

            else:
                # Forward
                SIR_all = markov_jump_process(x_init, p, u + V.T, t_step, T_max, settings_dict, sim_ID)
                adults = SIR_all[(0, 1, 2), :].T
                children = SIR_all[(3, 4, 5), :].T

                Jv = objective(adults, children,
                               stretch_vec(Us + V[0, :], Uw + V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

                # Backward
                SIR_all = markov_jump_process(x_init, p, u - V.T, t_step, T_max, settings_dict, sim_ID)
                adults = SIR_all[(0, 1, 2), :].T
                children = SIR_all[(3, 4, 5), :].T

                Jw = objective(adults, children,
                               stretch_vec(Us - V[0, :], Uw - V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

            Ju[j, k * duration_of_constant_control: (k + 1) * duration_of_constant_control] = (Jv - Jw) / fd_step_size

        # Compute last week (possibly incomplete)
        V = np.zeros((2, u.shape[0]))
        V[j, k_max - 1] = 0.5 * fd_step_size

        if np.any(u[:, 0] + V.T[:, 0] > u_school_max) or np.any(u[:, 1] + V.T[:, 1] > u_work_max):
            # Central
            SIR_all = markov_jump_process(x_init, p, u, t_step, T_max, settings_dict, sim_ID)
            adults = SIR_all[(0, 1, 2), :].T
            children = SIR_all[(3, 4, 5), :].T

            Jv = objective(adults, children,
                           stretch_vec(Us, Uw,
                                       total_timesteps, duration_of_constant_control), settings_dict)

            # Backward
            SIR_all = markov_jump_process(x_init, p, u - 2 * V.T, t_step, T_max, settings_dict, sim_ID)
            adults = SIR_all[(0, 1, 2), :].T
            children = SIR_all[(3, 4, 5), :].T

            Jw = objective(adults, children,
                           stretch_vec(Us - 2 * V[0, :], Uw - 2 * V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

        elif np.any(u[:, 0] + V.T[:, 0] > u_school_min) or np.any(u[:, 1] + V.T[:, 1] > u_work_min):
            # Forward
            SIR_all = markov_jump_process(x_init, p, u + 2 * V.T, t_step, T_max, settings_dict, sim_ID)
            adults = SIR_all[(0, 1, 2), :].T
            children = SIR_all[(3, 4, 5), :].T

            Jv = objective(adults, children,
                           stretch_vec(Us + 2 * V[0, :], Uw + 2 * V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

            # Central
            SIR_all = markov_jump_process(x_init, p, u, t_step, T_max, settings_dict, sim_ID)
            adults = SIR_all[(0, 1, 2), :].T
            children = SIR_all[(3, 4, 5), :].T

            Jw = objective(adults, children,
                           stretch_vec(Us, Uw,
                                       total_timesteps, duration_of_constant_control), settings_dict)

        else:
            # Forward
            SIR_all = markov_jump_process(x_init, p, u + V.T, t_step, T_max, settings_dict, sim_ID)
            adults = SIR_all[(0, 1, 2), :].T
            children = SIR_all[(3, 4, 5), :].T

            Jv = objective(adults, children,
                           stretch_vec(Us + V[0, :], Uw + V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

            # Backward
            SIR_all = markov_jump_process(x_init, p, u - V.T, t_step, T_max, settings_dict, sim_ID)
            adults = SIR_all[(0, 1, 2), :].T
            children = SIR_all[(3, 4, 5), :].T

            Jw = objective(adults, children,
                           stretch_vec(Us - V[0, :], Uw - V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

        Ju[j, (k_max - 1) * duration_of_constant_control::] = (Jv - Jw) / fd_step_size

    return Ju


# %%

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import json

    # Path to settings file
    path = '<path_to_input_and_output_directory>/opt_settings.txt'

    with open(path, 'r') as file:
        settings_dict = json.load(file)

    duration_of_constant_control = settings_dict['duration_of_constant_control']
    data_path = settings_dict['data_path']
    output_dir = settings_dict['output_directory']
    u_school_min = settings_dict['u_school_min']
    u_school_max = settings_dict['u_school_max']
    u_work_min = settings_dict['u_work_min']
    u_work_max = settings_dict['u_work_max']

    # Parameter setting
    T_max = (settings_dict['total_timesteps'] - 1) / 24
    t_step = 1 / 24

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

    # Get number of agents
    num_agents = sum(y0)

    # How many weeks to simulate
    upper_bound_week = math.ceil(settings_dict['total_timesteps'] / duration_of_constant_control)

    # Control
    Us = 0.0 * np.random.uniform(u_school_min, u_school_max, upper_bound_week)
    Uw = 0.0 * np.random.uniform(u_work_min, u_work_max, upper_bound_week)
    u = np.vstack((Us, Uw)).T

    # Run simulation
    start = time.time()
    trajectory = markov_jump_process(x_init,
                                     p,
                                     u,
                                     t_step,
                                     T_max,
                                     settings_dict,
                                     np.random.randint(0, 2**32))
    stop = time.time()

    simtime = np.linspace(0, T_max, trajectory.shape[1])

    fig = plt.figure()
    plt.step(simtime, trajectory[0, :]/num_agents, 'b', label=r'$S_\mathrm{a}$')
    plt.step(simtime, trajectory[1, :]/num_agents, 'r', label=r'$I_\mathrm{a}$')
    plt.step(simtime, trajectory[2, :]/num_agents, 'g', label=r'$R_\mathrm{a}$')
    plt.step(simtime, trajectory[3, :]/num_agents, 'b--', label=r'$S_\mathrm{c}$')
    plt.step(simtime, trajectory[4, :]/num_agents, 'r--', label=r'$I_\mathrm{c}$')
    plt.step(simtime, trajectory[5, :]/num_agents, 'g--', label=r'$R_\mathrm{c}$')
    plt.legend(ncol=2)
    plt.xlabel('Time [days]')
    plt.ylabel('Fraction of agents')

    print('\nElapsed time: %.4f seconds\n' % (stop - start))

    # Estimate gradient
    start = time.time()
    Ju = gradient(x_init, p, u, t_step, T_max, settings_dict, np.random.randint(0, 2**32))
    stop = time.time()
    print('Elapsed time: %.4f seconds\n' % (stop - start))

    print('Objective function gradient\n', condense_vec(Ju, duration_of_constant_control).T)

    adults = trajectory[(0, 1, 2), :].T
    children = trajectory[(3, 4, 5), :].T

    # Compute objective
    J = objective(adults, children, stretch_vec(Us, Uw, settings_dict['total_timesteps'], duration_of_constant_control), settings_dict)

    print('\nObjective function: %.2f' % J)
