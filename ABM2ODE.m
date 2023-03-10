clear all
close all

gen_samples = false;                                % Generate ABM samples?
save_fit = false;                                    % Save fitting parameter?
num_Monte_Carlo_simulations = 1000;

fileName = 'data/opt_settings.txt';                 % Filename
fid = fopen(fileName);                              % Opening the file
raw = fread(fid,inf);                               % Reading the contents
str = char(raw');                                   % Transformation
fclose(fid);                                        % Closing the file
data = jsondecode(str);                             % Using the jsondecode function to parse JSON from string

T_coarse = data.total_timesteps/24;                 % Coarse time, time units in [days]
T_fine = data.total_timesteps;                      % Fine time, time units in [hours]
N = data.total_timesteps;                           % Number of time steps
h_coarse = 1;                                       % Equidistant time step
h_fine = T_coarse/N;                                % Equidistant time step

u = zeros(2, N);

%% Run Python script

old_num_MC_sim = readmatrix(fullfile(data.data_path, 'num_MC_sim_obj_est.csv'));

if gen_samples
    num_MC_sim = num_Monte_Carlo_simulations;

    fileID = fopen(fullfile(data.data_path, 'num_MC_sim_obj_est.csv'), 'w');
    fprintf(fileID, num2str(num_MC_sim));

    system('chmod +x GERDA_state_estimation.py; ./GERDA_state_estimation.py');
    copyfile(strcat(data.data_path, '/adults_*'), fullfile(data.data_path, data.output_directory))
    copyfile(strcat(data.data_path, '/children_*'), fullfile(data.data_path, data.output_directory))
else
    disp('No data for fitting generated')
end

% Reset number of Monte Carlo simulations to previous
fileID = fopen(fullfile(data.data_path, 'num_MC_sim_obj_est.csv'), 'w');
fprintf(fileID, num2str(old_num_MC_sim));

%% Set up the Import Options and import the data

opts = delimitedTextImportOptions("NumVariables", 3);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["e02", "e00", "e00_1"];
opts.VariableTypes = ["double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
adultsstateestimate_mean = readtable(fullfile(data.data_path, data.output_directory, 'adults_state_estimate_mean.csv'), opts);
adultsstateestimate_var = readtable(fullfile(data.data_path, data.output_directory, 'adults_state_estimate_var.csv'), opts);
childrenstateestimate_mean = readtable(fullfile(data.data_path, data.output_directory, 'children_state_estimate_mean.csv'), opts);
childrenstateestimate_var = readtable(fullfile(data.data_path, data.output_directory, 'children_state_estimate_var.csv'), opts);

%% Convert to output type

adultsstateestimate_mean = table2array(adultsstateestimate_mean);
adultsstateestimate_var = table2array(adultsstateestimate_var);
childrenstateestimate_mean = table2array(childrenstateestimate_mean);
childrenstateestimate_var = table2array(childrenstateestimate_var);

%% ODE fit

% Initial state
y0 = [adultsstateestimate_mean(1, 1)
    childrenstateestimate_mean(1, 1)
    adultsstateestimate_mean(1, 2)
    childrenstateestimate_mean(1, 2)
    adultsstateestimate_mean(1, 3)
    childrenstateestimate_mean(1, 3)];

YmeanABM = [adultsstateestimate_mean(:, 1)'
    childrenstateestimate_mean(:, 1)'
    adultsstateestimate_mean(:, 2)'
    childrenstateestimate_mean(:, 2)'
    adultsstateestimate_mean(:, 3)'
    childrenstateestimate_mean(:, 3)'];

YvarABM = [adultsstateestimate_var(:, 1)'
    childrenstateestimate_var(:, 1)'
    adultsstateestimate_var(:, 2)'
    childrenstateestimate_var(:, 2)'
    adultsstateestimate_var(:, 3)'
    childrenstateestimate_var(:, 3)'];

% Least-squares fit with full observability
p_full = ODE_fit_full(y0, u, h_fine, YmeanABM, ones(size(YmeanABM)));

% Least-squares fit with partial observability (here only state I [infected agents])
p_partial = ODE_fit_partial(y0, u, h_fine, YmeanABM([3, 4], :), ones(size(YmeanABM([3, 4], :))));

% Save parameter
if save_fit
    writematrix(p_full, fullfile(data.data_path, data.output_directory, 'fitting_parameter.csv'))
    writematrix(y0, fullfile(data.data_path, data.output_directory, 'y0.csv'))
else
    disp('Fit not saved')
end

%% Plot

% Fine time grid
Y_fine = integrate_ODE(y0, p_full, u, h_fine);
Y_fine2 = integrate_ODE(y0, p_partial, u, h_fine);

% Coarse time grid
Y_coarse = integrate_ODE(y0, p_full, u(:, 1:24:end), h_coarse);

coarse_time = 0:h_coarse:T_coarse-1;
fine_time = linspace(0, T_coarse, N);

figure(1)
hold on
plot(coarse_time, Y_coarse/data.n_agents, 'r')
plot(fine_time, Y_fine/data.n_agents, 'k--')
legend('Coarse', 'Coarse', 'Coarse', 'Coarse', 'Coarse', 'Coarse', 'Fine', 'Fine', 'Fine', 'Fine', 'Fine', 'Fine')
hold off

figure()
subplot(3, 1, 1);
hold on
plot(fine_time, Y_fine(1:2, :)/data.n_agents, 'b')
plot(fine_time, Y_fine2(1:2, :)/data.n_agents, 'r')
plot(fine_time, adultsstateestimate_mean(:, 1)/data.n_agents, 'k')
plot(fine_time, childrenstateestimate_mean(:, 1)/data.n_agents, 'k')
title('S')
hold off

subplot(3, 1, 2);
hold on
plot(fine_time, Y_fine(3:4, :)/data.n_agents, 'b')
plot(fine_time, Y_fine2(3:4, :)/data.n_agents, 'r')
plot(fine_time, adultsstateestimate_mean(:, 2)/data.n_agents, 'k')
plot(fine_time, childrenstateestimate_mean(:, 2)/data.n_agents, 'k')
title('I')
hold off

subplot(3, 1, 3);
hold on
plot(fine_time, Y_fine(5:6, :)/data.n_agents, 'b')
plot(fine_time, Y_fine2(5:6, :)/data.n_agents, 'r')
plot(fine_time, adultsstateestimate_mean(:, 3)/data.n_agents, 'k')
plot(fine_time, childrenstateestimate_mean(:, 3)/data.n_agents, 'k')
title('R')
hold off

%% Integrate ODE

function [Y] = integrate_ODE(y0, p, u, h)
    N = size(u, 2);
    Y = zeros(length(y0), N);
    Y(:, 1) = y0;

    % Forward model - Runge kutta 4th order forward in time
    for i=1:N-1
        k1 = epidemic(Y(:, i), p, u(:, i));
        k2 = epidemic(Y(:, i) + h * k1/2, p, u(:, i));
        k3 = epidemic(Y(:, i) + h * k2/2, p, u(:, i));
        k4 = epidemic(Y(:, i) + h * k3,  p, u(:, i));
        Y(:, i + 1) = Y(:, i) + h/6 * (k1 + 2 * k2 + 2 * k3 + k4);
    end
end

function [Y] = integrate_ODE_partial(y0, p, u, h)   
    Y = integrate_ODE(y0, p, u, h);
    
    % Partial observability
    Y = Y([3, 4], :);
end

%% ODE

% Two-compartment SIRs model. This is the ODE's right hand side. y: state
% vector [Sa; Sc; Ia; Ic; Ra; Rc], where S is susceptible, I is infected, R is
% recovered, a denotes adults, c denotes children
% p: parameter vector [p1; p2; p3; p4; ra; rc] u: control vector [us; uo],
% where uh0 describes the home office fraction and us the school closing
% fraction fy: dy/dt
function [f, fy, fu] = epidemic(y, p, u)
    Sa = y(1); Sc = y(2); Ia = y(3); Ic = y(4);

    % Policies
    P1 = p(1) * (1 - u(2))^2;
    P2 = p(2) * (1 - 0.5 * u(1)) * (1 - 0.5 * u(2));
    P3 = p(3) * (1 - u(1))^2;

    % Derivative of policies wrt u
    P1_u = [ 0, -p(1) * 2 * (1 - u(2))];
    P2_u = [-0.5 * p(2) * (1 - 0.5 * u(2)), -0.5 * p(2) * (1 - 0.5 * u(1))];
    P3_u = [-2 * p(3) * (1 - u(1)), 0];

    % Compute transition rates between compartments
    Sa_Ia = Sa * (P1 * Ia + P2 * Ic);
    Sc_Ic = Sc * (P3 * Ic + P2 * Ia);
    Ia_Ra = p(4) * Ia;
    Ic_Rc = p(5) * Ic;

    % Compute derivatives of Sa_Ia, all others are zero
    Sa_Ia_Sa = P1 * Ia + P2 * Ic;
    Sa_Ia_Ia = Sa * P1;
    Sa_Ia_Ic = Sa * P2;

    % Compute derivatives of Sc_Ic, all others are zero
    Sc_Ic_Sc = P3 * Ic + P2 * Ia;
    Sc_Ic_Ia = Sc * P2;
    Sc_Ic_Ic = Sc * P3;

    % Compute derivatives of Ia_Ra, all others are zero
    Ia_Ra_Ia = p(4);

    % Compute derivatives of Ic_Rc, all others are zero
    Ic_Rc_Ic = p(5);

    % Compute derivatives wrt control u, all others are zero
    Sa_Ia_u = Sa * (P1_u * Ia + P2_u * Ic);
    Sc_Ic_u = Sc * (P3_u * Ic + P2_u * Ia);

    f = [-Sa_Ia;
         -Sc_Ic;
         Sa_Ia - Ia_Ra;
         Sc_Ic - Ic_Rc;
         Ia_Ra;
         Ic_Rc];

    fy = [-Sa_Ia_Sa 0         -Sa_Ia_Ia          -Sa_Ia_Ic          0 0
          0         -Sc_Ic_Sc -Sc_Ic_Ia          -Sc_Ic_Ic          0 0
          Sa_Ia_Sa 0           Sa_Ia_Ia-Ia_Ra_Ia  Sa_Ia_Ic          0 0
          0          Sc_Ic_Sc  Sc_Ic_Ia           Sc_Ic_Ic-Ic_Rc_Ic 0 0
          0         0          Ia_Ra_Ia           0                 0 0
          0         0          0                  Ic_Rc_Ic          0 0 ];

    fu = [-Sa_Ia_u
          -Sc_Ic_u
          Sa_Ia_u
          Sc_Ic_u
          0 0
          0 0 ];
end

%%

function [p_full] = ODE_fit_full(y0, u, h, YmeanABM, YvarABM)

    YvarABM(YvarABM==0) = 1e-9;
    r = optimvar('r', 5, "LowerBound", 1e-4, "UpperBound", 0.1);
    r.LowerBound(1) = 1e-4;
    r.LowerBound(3) = 1e-4;

    optifun = fcn2optimexpr(@integrate_ODE, y0, r, u, h);
    obj = sum(sum(((optifun - YmeanABM).^2)./YvarABM));

    prob = optimproblem('ObjectiveSense', 'minimize', 'Objective', obj);

    % Settings, 3000 is default
    options = optimoptions(prob, 'MaxFunctionEvaluations', 10000);

    % Initial value for nonlinear problems, magnitudes already correct
    r0.r = [1e-6, 1e-4, 1e-4, 1e-2, 1e-2];

    [rsol, ~] = solve(prob, r0, 'Options', options);
    disp(rsol.r);
    p_full = rsol.r;

end

function [p_full] = ODE_fit_partial(y0, u, h, YmeanABM, YvarABM)

    YvarABM(YvarABM==0) = 1e-9;
    r = optimvar('r', 5, "LowerBound", 1e-4, "UpperBound", 0.1);
    r.LowerBound(1) = 1e-4;
    r.LowerBound(3) = 1e-4;

    optifun = fcn2optimexpr(@integrate_ODE_partial, y0, r, u, h);
    obj = sum(sum(((optifun - YmeanABM).^2)./YvarABM));

    prob = optimproblem('ObjectiveSense', 'minimize', 'Objective', obj);

    % Settings, 3000 is default
    options = optimoptions(prob, 'MaxFunctionEvaluations', 10000);

    % Initial value for nonlinear problems, magnitudes already correct
    r0.r = [1e-6, 1e-4, 1e-4, 1e-2, 1e-2];

    [rsol, ~] = solve(prob, r0, 'Options', options);
    disp(rsol.r);
    p_full = rsol.r;

end
