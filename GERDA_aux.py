#!<path_to_GERDA_virtuel_environment>/envs/gerdaenv/bin/python3.8
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:42:55 2022

Implements the GERDA model as needed for the optimization process

@author: Jan-Hendrik Niemann
"""


import os
import numpy as np
import pandas as pd
from scipy import integrate
import math
import random
import json
import copy
import pickle
import sys

# Path to json file containing settings
path = '<path_to_input_and_output_directory>/opt_settings.txt'

# Load settings
with open(path, 'r') as file:
    settings_dict = json.load(file)
    path_to_GERDA = settings_dict['path_to_GERDA']
    settings_dict = None

# Change current working directory (required for MATLAB)
owd = os.getcwd()
os.chdir(path_to_GERDA)

# Needs to be here for MATLAB compatibility
from gerda.core.virusPropagationModel import ModeledPopulatedWorld, Simulation


# %%

def baseline_scenario(modeledWorld, Us, Uw, settings_dict, sim_id=None):
    """
    Simulates GERDA given the controls Us (home schooling) and Uw (home office)

    Parameters
    ----------
    modeledWorld : GERDA world object
        See documentation of GERDA.
    Us : ndarray
        Control vector for home schooling.
    Uw : ndarray
        Control vector for home office.
    settings_dict : dict
        Dictionary with optimization settings.
    sim_ID : int
        Unique simulation identifier. Serves also as seed. The default is None.

    Returns
    -------
    df_adults : DataFrame
        SIR time course of of age agents.
    df_children : DataFrame
        SIR time course of underage agents.

    """

    def reset_sim_settings():
        for p in simulation.people:
            p.reset_schedule()

    def control_implmt_func(us, Uw):
        prob_dict = {'work': Uw, 'school': us}

        for p in simulation.people:
            for i, loc in enumerate(p.schedule['locs']):
                prob = random.random()
                if loc.location_type in prob_dict:
                    if prob < prob_dict[loc.location_type]:
                        # If child is less then 13 years old, a supervisor is
                        # needed at home
                        if p.age <= 12:
                            # Assign child to its home
                            p.schedule['locs'][i] = p.home
                            # Find the corresponding supervisor (not necessary
                            # a patent but an adult who lives in the same household)
                            for x in simulation.people:
                                if x.age > 18:
                                    if x.home.ID == p.home.ID:
                                        # Assign adult to its home
                                        x.schedule['locs'][i] = x.home

                        else:
                            p.schedule['locs'][i] = p.home

    # Set simulation ID
    if sim_id is None:
        sim_id = np.random.randint(0, 2**32 - 1)

    # Load settings
    total_timesteps = settings_dict['total_timesteps']
    duration_of_constant_control = settings_dict['duration_of_constant_control']

    # Set up simulation
    simulation = Simulation(modeledWorld,
                            duration_of_constant_control,
                            sim_ID=sim_id,
                            run_immediately=False,
                            copy_sim_object=True)
    simulation.change_agent_attributes({'all': {'behaviour_as_infected': {'value': settings_dict['general_infectivity'], 'type': 'multiplicative_factor'}}})

    # Set interaction frequency
    simulation.interaction_frequency = settings_dict['general_interaction_frequency']

    # Run simulation
    sim_time = 0
    week_num = 0
    while sim_time < total_timesteps:
        # Complete weeks
        if (total_timesteps - week_num * duration_of_constant_control) >= duration_of_constant_control:
            simulation.time_steps = duration_of_constant_control
            sim_time = sim_time + duration_of_constant_control
        # Incomplete week
        else:
            simulation.time_steps = total_timesteps - week_num * duration_of_constant_control
            sim_time = sim_time + total_timesteps - week_num * duration_of_constant_control

        # Set controls
        control_implmt_func(Us[week_num], Uw[week_num])

        # Simulate ABM
        simulation.simulate(return_timecourse=False)

        # Reset schedule
        reset_sim_settings()
        week_num = week_num + 1

    # Split dataframe to "children" and "adults"
    pd.options.mode.chained_assignment = None  # Turn off warnings
    df_adults = simulation.get_status_trajectories_for_specific_state(state='infection',
                                                                      filter=('age', np.arange(19, 190)),
                                                                      max_time=total_timesteps - 1,
                                                                      filtered_for_state=False)

    df_children = simulation.get_status_trajectories_for_specific_state(state='infection',
                                                                        filter=('age', np.arange(0, 19)),
                                                                        max_time=total_timesteps - 1,
                                                                        filtered_for_state=False)

    return df_adults, df_children


def stretch_vec(Us, Uw, total_timesteps, duration_of_constant_control):
    """
    Auxiliary function to map control vectors Us and Uw from weekly to hourly

    Parameters
    ----------
    Us : ndarray
        Control vector home schooling.
    Uw : ndarray
        Control vector home office.
    total_timesteps : int, optional
        Number of time steps in hours.
    duration_of_constant_control : int, optional
        Duration in hours for a constant control.

    Returns
    -------
    u : ndarray
        Mapped control vector as 2 x time with u[0, :] = Us(t), u[1, :] = Uw(t)

    """

    # Allocate memory
    u = np.zeros((2, total_timesteps))

    sim_time = 0
    week_num = 0
    while sim_time < total_timesteps:
        # Complete weeks
        if (total_timesteps - week_num * duration_of_constant_control) >= duration_of_constant_control:
            u[0, week_num * duration_of_constant_control: (week_num + 1) * duration_of_constant_control] = Us[week_num]
            u[1, week_num * duration_of_constant_control: (week_num + 1) * duration_of_constant_control] = Uw[week_num]
            sim_time = sim_time + duration_of_constant_control
        # Incomplete week
        else:
            u[0, week_num * duration_of_constant_control:] = Us[week_num]
            u[1, week_num * duration_of_constant_control:] = Uw[week_num]
            sim_time = sim_time + total_timesteps - week_num * duration_of_constant_control

        week_num = week_num + 1

    return u


def condense_vec(u, duration_of_constant_control):
    """
    Auxiliary function to map control matrix u from hourly to weekly

    Parameters
    ----------
    u : ndarray
        Two dimensional array.
    duration_of_constant_control : int
        Keep every duration_of_constant_control-th value in array.

    Returns
    -------
    U : ndarray
        Array with same number of rows but reduced columns.

    """

    U = u[:, ::duration_of_constant_control]

    return U


def objective(adults, children, U, settings_dict):
    """
    Evaluate objective function

    Parameters
    ----------
    adults : DataFrame
        GERDA output for adults.
    children : DataFrame
        GERDA output for children (underage).
    U : ndarray
        Control matrix with dimension 2 x time with U[0, :] = Us(t),
        U[1, :] = Uw(t).
    settings_dict : dict
        Dictionary with optimization settings.

    Returns
    -------
    J_int : float
        Value of objective function.

    """

    # Weights for social impact
    so = 1
    # Weights for economial impact
    ho = 1
    imax = 0.005
    u_work_max = settings_dict['u_work_max'] + 0.1

    # Time steps in days
    total_timesteps = settings_dict['total_timesteps'] / 24

    # Allocate memory
    J = np.zeros(len(adults))

    # Evaluation if data is given as Pandas data frame
    if isinstance(adults, pd.DataFrame):
        num_people = adults.iloc[0].sum() + children.iloc[0].sum()

        for i in range(len(adults)):

            # Infected adults/children
            Ia = adults.I[i]
            Ic = children.I[i]

            # Control
            u = U[:, i]

            if u[1] >= u_work_max:
                J[i] = np.inf
            else:
                J[i] = ((Ia + Ic) / num_people
                        + math.exp(10 * ((Ia + Ic) - imax * num_people) / num_people)
                        + so * u[0]**2 - ho * math.log(u_work_max - u[1]))

    # Evaluation as numpy array
    else:
        num_people = np.sum(adults[0, :]) + np.sum(children[0, :])

        for i in range(adults.shape[0]):

            # Infected adults/children
            Ia = adults[i, 1]
            Ic = children[i, 1]

            # Control
            u = U[:, i]

            if u[1] >= u_work_max:
                J[i] = np.inf
            else:
                J[i] = ((Ia + Ic) / num_people
                        + math.exp(10 * ((Ia + Ic) - imax * num_people) / num_people)
                        + so * u[0]**2 - ho * math.log(u_work_max - u[1]))

    J_int = integrate.simpson(J, np.linspace(0, total_timesteps, len(J)))

    return J_int


def gradient(modeledWorld, Us, Uw, settings_dict, sim_ID):
    """
    Compute the gradient of the objective function at control u via finite
    differences

    Parameters
    ----------
    modeledWorld : GERDA world object
        See documentation of GERDA.
    Us : ndarray
        Control vector for home schooling.
    Uw : ndarray
        Control vector for home office.
    settings_dict : dict
        Dictionary with optimization settings.
    sim_ID : int
        Unique simulation identifier. The default is None.

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

    # Number of complete weeks
    k_max = len(Us)

    j = 0
    k = 0

    for j in range(Ju.shape[0]):
        for k in range(k_max - 1):

            V = np.zeros((2, len(Us)))
            V[j, k] = 0.5 * fd_step_size

            if np.any(Us + V[0, :] > u_school_max) or np.any(Uw + V[1, :] > u_work_max):
                # Central
                sim_id = int(str(sim_ID) + str(j) + str(k) + '0')
                adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                     Us,
                                                     Uw,
                                                     settings_dict,
                                                     sim_id)

                Jv = objective(adults, children,
                               stretch_vec(Us, Uw,
                                           total_timesteps, duration_of_constant_control), settings_dict)

                # Backward
                sim_id = int(str(sim_ID) + str(j) + str(k) + '1')
                adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                     Us - 2 * V[0, :],
                                                     Uw - 2 * V[1, :],
                                                     settings_dict,
                                                     sim_id)

                Jw = objective(adults, children,
                               stretch_vec(Us - 2 * V[0, :], Uw - 2 * V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

            elif np.any(Us - V[0, :] < u_school_min) or np.any(Uw - V[1, :] < u_work_min):
                # Forward
                sim_id = int(str(sim_ID) + str(j) + str(k) + '2')
                adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                     Us + 2 * V[0, :],
                                                     Uw + 2 * V[1, :],
                                                     settings_dict,
                                                     sim_id)

                Jv = objective(adults, children,
                               stretch_vec(Us + 2 * V[0, :], Uw + 2 * V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

                # Central
                sim_id = int(str(sim_ID) + str(j) + str(k) + '0')
                adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                     Us,
                                                     Uw,
                                                     settings_dict,
                                                     sim_id)

                Jw = objective(adults, children,
                               stretch_vec(Us, Uw,
                                           total_timesteps, duration_of_constant_control), settings_dict)

            else:
                # Forward
                sim_id = int(str(sim_ID) + str(j) + str(k) + '2')
                adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                     Us + V[0, :],
                                                     Uw + V[1, :],
                                                     settings_dict,
                                                     sim_id)

                Jv = objective(adults, children,
                               stretch_vec(Us + V[0, :], Uw + V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

                # Backward
                sim_id = int(str(sim_ID) + str(j) + str(k) + '1')
                adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                     Us - V[0, :],
                                                     Uw - V[1, :],
                                                     settings_dict,
                                                     sim_id)

                Jw = objective(adults, children,
                               stretch_vec(Us - V[0, :], Uw - V[1, :],
                                           total_timesteps, duration_of_constant_control), settings_dict)

            Ju[j, k * duration_of_constant_control: (k + 1) * duration_of_constant_control] = (Jv - Jw) / fd_step_size

        # Compute last week (possibly incomplete)
        V = np.zeros((2, len(Us)))
        V[j, k_max - 1] = 0.5 * fd_step_size

        if np.any(Us + V[0, :] > u_school_max) or np.any(Uw + V[1, :] > u_work_max):
            # Central
            sim_id = int(str(sim_ID) + str(j) + str(k) + '0')
            adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                 Us,
                                                 Uw,
                                                 settings_dict,
                                                 sim_id)

            Jv = objective(adults, children,
                           stretch_vec(Us, Uw,
                                       total_timesteps, duration_of_constant_control), settings_dict)

            # Backward
            sim_id = int(str(sim_ID) + str(j) + str(k) + '1')
            adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                 Us - 2 * V[0, :],
                                                 Uw - 2 * V[1, :],
                                                 settings_dict,
                                                 sim_id)

            Jw = objective(adults, children,
                           stretch_vec(Us - 2 * V[0, :], Uw - 2 * V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

        elif np.any(Us - V[0, :] < u_school_min) or np.any(Uw - V[1, :] < u_work_min):
            # Forward
            sim_id = int(str(sim_ID) + str(j) + str(k) + '2')
            adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                 Us + 2 * V[0, :],
                                                 Uw + 2 * V[1, :],
                                                 settings_dict,
                                                 sim_id)

            Jv = objective(adults, children,
                           stretch_vec(Us + 2 * V[0, :], Uw + 2 * V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

            # Central
            sim_id = int(str(sim_ID) + str(j) + str(k) + '0')
            adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                 Us,
                                                 Uw,
                                                 settings_dict,
                                                 sim_id)

            Jw = objective(adults, children,
                           stretch_vec(Us, Uw,
                                       total_timesteps, duration_of_constant_control), settings_dict)

        else:
            # Forward
            sim_id = int(str(sim_ID) + str(j) + str(k) + '2')
            adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                 Us + V[0, :],
                                                 Uw + V[1, :],
                                                 settings_dict,
                                                 sim_id)

            Jv = objective(adults, children,
                           stretch_vec(Us + V[0, :], Uw + V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

            # Backward
            sim_id = int(str(sim_ID) + str(j) + str(k) + '1')
            adults, children = baseline_scenario(copy.deepcopy(modeledWorld),
                                                 Us - V[0, :],
                                                 Uw - V[1, :],
                                                 settings_dict,
                                                 sim_id)

            Jw = objective(adults, children,
                           stretch_vec(Us - V[0, :], Uw - V[1, :],
                                       total_timesteps, duration_of_constant_control), settings_dict)

        Ju[j, (k_max - 1) * duration_of_constant_control::] = (Jv - Jw) / fd_step_size

    return Ju


def load(settings_dict, create_new_world=True):
    """
    Load or create a new GERDA world object

    Parameters
    ----------
    settings_dict : dict
        Dictionary with optimization settings.
    create_new_world : bool, optional
        Create a new GERDA world object in case no other exists. The default is
        True.

    Returns
    -------
    modeledWorld : GERDA world object
        See documentation of GERDA.

    """

    # Load settings
    data_path = settings_dict['data_path']
    output_directory = settings_dict['output_directory']
    saved_world = settings_dict['saved_world']
    n_initially_infected = settings_dict['n_initially_infected']

    # Load world
    try:
        with open(os.path.join(data_path, output_directory, saved_world), 'rb') as file:
            modeledWorld = pickle.load(file)
            print('Loaded world: %s' % saved_world)

    except OSError:
        print(r'Could not find world "%s"' % saved_world)
        if create_new_world:
            geopath = settings_dict['geopath']
            geofiles = settings_dict['geofiles']

            # Conversion of string keys of numeric keys
            geofiles = {int(k): v for k, v in geofiles.items()}
            world_to_pick = settings_dict['world_to_pick']

            modeledWorld = ModeledPopulatedWorld(initial_infections=n_initially_infected,
                                                 geofile_name=geopath+geofiles[world_to_pick],
                                                 input_schedules='schedules_v2')

            # Get number of agents
            n_people = modeledWorld.number_of_people

            # Save world
            modeledWorld.save('Reduced_Gangelt_n' + str(n_people), folder=os.path.join(data_path, output_directory, ''))

            new_world_name = 'Reduced_Gangelt_n' + str(n_people) + '_worldObj.pkl'
            print('Created new world with name: %s' % new_world_name)

            # Update settings file
            settings_dict['saved_world'] = new_world_name
            settings_dict['n_agents'] = n_people
            with open(os.path.join(data_path, 'opt_settings.txt'), 'w') as json_file:
                json.dump(settings_dict, json_file, indent=4, sort_keys=True, ensure_ascii=True)
            print('Updated opt_settings.txt with new world')

        else:
            sys.exit()

    return modeledWorld


def epidemic_ODE(t, y, p, u):
    """
    Right-hand side of ODE. 2-compartment SIR model

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

    # Transition rates between compartments
    Sa_Ia = Sa * (P1 * Ia + P2 * Ic)
    Sc_Ic = Sc * (P3 * Ic + P2 * Ia)
    Ia_Ra = p[3] * Ia
    Ic_Rc = p[4] * Ic

    f = np.array([-Sa_Ia,
                  -Sc_Ic,
                  Sa_Ia - Ia_Ra,
                  Sc_Ic - Ic_Rc,
                  Ia_Ra,
                  Ic_Rc])

    return f


def gradient_projection(du, u, settings_dict):
    """
    Projection of gradient onto feasible space

    Parameters
    ----------
    du : ndarray
        Descent direction.
    u : ndarray
        Control vector u = [u_school, u_homeoffice].
    settings_dict : dict
        Dictionary with optimization settings.

    Raises
    ------
    SystemExit
        Error if control u and descent direction du have different shapes.

    Returns
    -------
    du : ndarray
        Projected descent direction.

    """

    # Load settings
    u_school_min = settings_dict['u_school_min']
    u_school_max = settings_dict['u_school_max']
    u_work_min = settings_dict['u_work_min']
    u_work_max = settings_dict['u_work_max']

    if du.shape != du.shape:
        raise SystemExit('Control u and descent direction du have different shapes')

    for i in range(u.shape[0]):
        if (u[i, 0] <= u_school_min and du[i, 0] < 0) or (u[i, 0] >= u_school_max and du[i, 0] > 0):
            du[i, 0] = 0
        if (u[i, 1] <= u_work_min and du[i, 1] < 0) or (u[i, 1] >= u_work_max and du[i, 1] > 0):
            du[i, 1] = 0

    return du


# %%

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time

    # Path to settings file
    path = os.path.join(owd, 'data/opt_settings.txt')

    # If no saved world is found, create new and save it
    create_new_world = True

    with open(path, 'r') as file:
        settings_dict = json.load(file)

    # Load world
    modeledWorld = load(settings_dict, create_new_world)

    # Load settings
    total_timesteps = settings_dict['total_timesteps']
    duration_of_constant_control = settings_dict['duration_of_constant_control']
    n_agents = settings_dict['n_agents']
    data_path = settings_dict['data_path']
    output_dir = settings_dict['output_directory']
    u_school_min = settings_dict['u_school_min']
    u_school_max = settings_dict['u_school_max']
    u_work_min = settings_dict['u_work_min']
    u_work_max = settings_dict['u_work_max']

    # How many weeks to simulate
    upper_bound_week = math.ceil(total_timesteps / duration_of_constant_control)

    Us = 0.0 * np.random.uniform(u_school_min, u_school_max, upper_bound_week)
    Uw = 0.0 * np.random.uniform(u_work_min, u_work_max, upper_bound_week)

    tic = time.time()
    adults, children = baseline_scenario(copy.deepcopy(modeledWorld), Us, Uw, settings_dict)

    print('\nElapsed time %.4f seconds\n' % (time.time() - tic))

    time_axis = np.linspace(0, total_timesteps/24, total_timesteps)

    fig = plt.figure()
    plt.step(time_axis, adults.S/n_agents, 'b', label='$S_a$', where='post')
    plt.step(time_axis, adults.I/n_agents, 'r', label='$I_a$', where='post')
    plt.step(time_axis, adults.R/n_agents, 'g', label='$R_a$', where='post')
    plt.step(time_axis, children.S/n_agents, 'b--', label='$S_c$', where='post')
    plt.step(time_axis, children.I/n_agents, 'r--', label='$I_c$', where='post')
    plt.step(time_axis, children.R/n_agents, 'g--', label='$R_c$', where='post')
    plt.legend()
    plt.xlabel('Time [days]')
    plt.ylabel('Fraction of agents')
    plt.savefig(os.path.join(data_path, output_dir, 'GERDA_trajectory.png'))

    fig = plt.figure()
    plt.step(np.linspace(0, total_timesteps/24, upper_bound_week + 1), np.hstack((Us, Us[-1])), where='post')
    plt.step(np.linspace(0, total_timesteps/24, upper_bound_week + 1), np.hstack((Uw, Uw[-1])), where='post')
    plt.xlabel('Time [days]')
    plt.ylabel('Control $u(t)$')
    plt.legend(('$u_s(t)$', '$u_w(t)$'))
    plt.ylim((-0.05, 1.05))

    # Compute objective
    J = objective(adults, children, stretch_vec(Us, Uw, total_timesteps, duration_of_constant_control), settings_dict)

    print('Objective function: %.2f' % J)
