clear all
close all

% Load json
fileName = "data/opt_settings.txt";
fid = fopen(fileName, "r");
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
data = jsondecode(str);

% Readme file with settings
fileID = fopen(fullfile(data.data_path, data.output_directory, "README.txt"), "w");
if data.fast_mode
    fprintf(fileID, "Multilevel Optimization fast dummy ABM\n\n");
else
    fprintf(fileID, "Multilevel Optimization dummy ABM\n\n");
end
fns = fieldnames(data);
for i=1:length(fns)
    switch class(data.(fns{i}))
        case 'char'
            fprintf(fileID, "%s: %s\n", fns{i}, data.(fns{i}));
        case 'struct'
            fns_tmp = fieldnames(data.(fns{i}));
            for j=1:length(fns_tmp)
                fprintf(fileID, "%s world %i: %s\n", fns{i}, j, data.(fns{i}).(fns_tmp{j}));
            end
        otherwise
            fprintf(fileID, "%s: %f\n", fns{i}, data.(fns{i}));
    end
end
fclose(fileID);

% Load settings
T_coarse = data.total_timesteps / 24;                                          % Coarse time, time units in [days]
N = T_coarse;                                                                  % Number of time steps
h = T_coarse / N;                                                              % Equidistant time step
t = 0:h:T_coarse;                                                              % Time domain
duration_of_constant_control = data.duration_of_constant_control / 24;         % Unit in [days]
fd_step_size = data.fd_step_size;                                              % Finite difference step size
itermax = data.max_iterations;                                                 % Maximum iterations of multi-level algorithm
fit_ODE = data.fit_ODE;                                                        % ﻿Fit ODE in every iteration
u_school_min = data.u_school_min;
u_school_max = data.u_school_max;
u_work_min = data.u_work_min;
u_work_max = data.u_work_max;

% Set up control
upper_bound_weeks = ceil(T_coarse / duration_of_constant_control);
u = zeros(2, upper_bound_weeks);                                               % Initial guess
u(2, u(2, :) > u_work_max) = u_work_max;                                       % Upper bound for u
umin = zeros(2, upper_bound_weeks);                                            % Lower bound for u
umax = [u_school_max; u_work_max] * ones(1, upper_bound_weeks);                % Upper bound for u

% GERDA fitted parameter
p = readmatrix(fullfile(data.data_path, data.output_directory, "fitting_parameter.csv"));

% Initial state of fitted coarse model
y0 = readmatrix(fullfile(data.data_path, data.output_directory, "y0.csv"));

% Current number of agents
n_agents = sum(y0);

% Define function handles
fEpi = @epidemic;
fObj = @objective;
estimate_objective = true;
estimate_gradient = true;
fineF = @(u, estimate_objective, estimate_gradient) fine(u, estimate_objective, estimate_gradient, data);

% Allocate memory
J_vec = NaN(1, itermax + 1);
ABM_evaluations_vec = zeros(1, itermax + 1);
p_vec = zeros(length(p), itermax);

%% Run optimization

% Reset number of Monte Carlo samples
fileID = fopen(fullfile(data.data_path, "num_MC_sim_obj_est.csv"), "w");
fprintf(fileID, num2str(data.obj_est_MC_sim_init));
fileID = fopen(fullfile(data.data_path, "num_MC_sim_grad_est.csv"), "w");
fprintf(fileID, num2str(data.grad_est_MC_sim_init));

% Initial fine simulation with u controls
fileID = fopen(fullfile(data.data_path, "alpha.csv"), "w");
fprintf(fileID, num2str(1));
fileID = fopen(fullfile(data.data_path, "rho.csv"), "w");
fprintf(fileID, num2str(fd_step_size));
fileID = fopen(fullfile(data.data_path, "error_e.csv"), "w");
fprintf(fileID, num2str(0.0));
[J, Ju, num_MC_sim_obj_est, num_MC_sim_grad_est, eps] = fineF(u, true, true);
J_vec(1) = J;
ABM_evaluations_vec(1) = num_MC_sim_obj_est + 4 * num_MC_sim_grad_est * size(Ju, 2);

% Save as csv files
writematrix(Ju', fullfile(data.data_path, data.output_directory, "Ju_0.csv"));
writematrix(u', fullfile(data.data_path, data.output_directory, "u_coarse_0.csv"));

% Coarse model
Coarse.objective = @(u) reducedObjective(y0, p, u, h, fEpi, fObj, N, duration_of_constant_control);
Coarse.project = @(u) u;
Coarse.prolongate = @(du) du;
Coarse.restrict = @(Ju) Ju;

tic
for iter = 1:itermax
    fprintf("\n+ + + Iteration No. %d of %d at %s + + +\n", iter, itermax, datetime('now', 'Format', 'HH:mm:ss'));

    if fit_ODE
        % Set up the Import Options and import the data
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
        adultsstateestimate_mean = readtable(fullfile(data.data_path, data.output_directory, 'adults_state_estimate_mean_tmp.csv'), opts);
        adultsstateestimate_var = readtable(fullfile(data.data_path, data.output_directory, 'adults_state_estimate_var_tmp.csv'), opts);
        childrenstateestimate_mean = readtable(fullfile(data.data_path, data.output_directory, 'children_state_estimate_mean_tmp.csv'), opts);
        childrenstateestimate_var = readtable(fullfile(data.data_path, data.output_directory, 'children_state_estimate_var_tmp.csv'), opts);

        % Convert to output type
        adultsstateestimate_mean = table2array(adultsstateestimate_mean);
        adultsstateestimate_var = table2array(adultsstateestimate_var);
        childrenstateestimate_mean = table2array(childrenstateestimate_mean);
        childrenstateestimate_var = table2array(childrenstateestimate_var);

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
        u_long = stretch_vec(u(1, :), u(2, :), data.total_timesteps, data.duration_of_constant_control);
        p = ODE_fit_full(y0, u_long, h, YmeanABM, ones(size(YmeanABM)));

        % Save parameter p
        p_vec(:, iter) = p;
        writematrix(p_vec, fullfile(data.data_path, data.output_directory, "p_vec.csv"));

        Coarse.objective = @(u) reducedObjective(y0, p, u, h, fEpi, fObj, N, duration_of_constant_control);

        Y_fine = integrate_ODE(y0, p, u_long, h);
        fine_time = linspace(0, T_coarse, length(Y_fine));

        figure(4)
        clf(4)
        hold on
        plt_fit = plot(fine_time, Y_fine / n_agents, 'r', 'linewidth', 2);
        plt_true = plot(fine_time, adultsstateestimate_mean' / n_agents, 'k--', 'linewidth', 2);
        plot(fine_time, childrenstateestimate_mean' / n_agents, 'k--', 'linewidth', 2)
        hold off
        ax=gca;
        ax.FontSize=12;
        legend([plt_fit(1) plt_true(1)], {'ODE', 'ABM'})
        xlabel("Time [days]")
        ylabel("Fraction of agents")
        savefig(fullfile(data.data_path, data.output_directory, strcat("ODEfit_", int2str(iter), ".fig")))
    end

    Jold = J;
    u_old = u;

    [u_new, J, Ju, num_MC_sim_obj_est, num_MC_sim_grad_est, ABM_evaluations] = recursiveTrustRegionStep(fineF, Coarse, u, J, Ju, umin, umax, num_MC_sim_obj_est, num_MC_sim_grad_est, data, iter);

    mask = (Ju>0) .* (u_new==umin) + (Ju<0) .* (u_new==umax);
    Ju = Ju .* (mask==0);

    fprintf("J_old = %f\nJ_new = %f\n|dJ| = %f\ndecrease = %f\n|delta_u| = %f\n", Jold, J, norm(Ju(:)), J - Jold, norm((u_new - u)));

    u = u_new;
    J_vec(iter + 1) = J;
    ABM_evaluations_vec(iter + 1) = ABM_evaluations;

    fprintf("ABM evaluations: %d\n", max(cumsum(ABM_evaluations_vec)));

    % Save as csv files
    writematrix(J_vec', fullfile(data.data_path, data.output_directory, "J_vec.csv"));
    writematrix(Ju', fullfile(data.data_path, data.output_directory, strcat("Ju_", int2str(iter), ".csv")));
    writematrix(u', fullfile(data.data_path, data.output_directory, strcat("u_coarse_", int2str(iter), ".csv")));
    writematrix(ABM_evaluations_vec', fullfile(data.data_path, data.output_directory, "ABM_evaluations_vec.csv"));

    % Reset finite difference step size
    fileID = fopen(strcat(data.data_path, "/rho.csv"), "w");
    fprintf(fileID, num2str(fd_step_size));

    % Plot control
    figure(1)
    U_new = stretch_vec(u_new(1, :), u_new(2, :), T_coarse, duration_of_constant_control);
    stairs(t(1:end-1), U_new', 'linewidth', 2);
    title("Multilevel scheme")
    legend("u_s(t)","u_h(t)")
    xlabel("Time [days]")
    ylabel("Control u(t)")
    ylim([0 1])
    ax=gca;
    ax.FontSize=12;
    savefig(fullfile(data.data_path, data.output_directory, "control.fig"))

    % Plot objective
    figure(2)
    plot(0:1:itermax, J_vec, 'linewidth', 2)
    xlabel("Iterations")
    ylabel("J(u)")
    ax=gca;
    ax.FontSize=12;
    savefig(fullfile(data.data_path, data.output_directory, "objective.fig"))

    % No further progress can be expected
    if u_old == u_new
        fprintf("\nIteration stopped as no further progress can be expected...\n\n")
        break
    end

end
toc

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
adultsstateestimate_mean = readtable(fullfile(data.data_path, data.output_directory, "adults_state_estimate_mean_tmp.csv"), opts);
childrenstateestimate_mean = readtable(fullfile(data.data_path, data.output_directory, "children_state_estimate_mean_tmp.csv"), opts);

%% Convert to output type

adultsstateestimate_mean = table2array(adultsstateestimate_mean);
childrenstateestimate_mean = table2array(childrenstateestimate_mean);

%% Plot result with optimal control

fine_time = linspace(0, data.total_timesteps / 24, data.total_timesteps);

figure(3)
hold on
plot(fine_time, adultsstateestimate_mean(:, 1)/n_agents, 'b', 'linewidth', 2)
plot(fine_time, childrenstateestimate_mean(:, 1)/n_agents, 'b', 'linewidth', 2)
plot(fine_time, adultsstateestimate_mean(:, 2)/n_agents, 'r', 'linewidth', 2)
plot(fine_time, childrenstateestimate_mean(:, 2)/n_agents, 'r', 'linewidth', 2)
plot(fine_time, adultsstateestimate_mean(:, 3)/n_agents, 'g', 'linewidth', 2)
plot(fine_time, childrenstateestimate_mean(:, 3)/n_agents, 'g', 'linewidth', 2)
hold off
xlabel("Time [days]")
ylabel("Fraction of agents")
savefig(fullfile(data.data_path, data.output_directory, "trajectory_with_final_control.fig"))

%% ------------------------------------------------------------------------

function [u, J, Ju, num_MC_sim_obj_est, num_MC_sim_grad_est, ABM_evaluations] = recursiveTrustRegionStep(fine_fct, Coarse, u, J, Ju, umin, umax, num_MC_sim_obj_est, num_MC_sim_grad_est, data, itr)
    % Count fine model evaluations in this step
    ABM_evaluations = 0;

    % Initial trust region radius
    rho = data.rho;

    % Maximum number of coarse optimization steps
    coarse_steps = data.max_coarse_steps;

    % Bounds on controls
    umin = max(umin, u - rho);
    umax = min(umax, u + rho);

    % Initial coasre trust region radius
    rhoCoarse = rho;
    rhoCoarsePersist = rho;

    iter = 0;

    % Do until fine step accepted
    accepted = false;
    counter_filename = 0;
    while ~accepted
        fprintf("\nRTRS No. %d: \n", iter + 1);
        % First do a coarse step.

        % Minimize coarse model over trust region -> trial point
        % First project to current iterate to the coarse space
        uCoarse = Coarse.project(u);
        % And restrict the objective's derivative to the coarse space
        JuCoarseRes = Coarse.restrict(Ju);

        % Check if the restricted gradient is sufficiently large - only then
        % a coarse correction can be expected to provide significant progress.
        % Otherwise, just do a fine level trust region step.

        % Correct the coarse model additively to first order consistency
        [~, JuCoarse] = Coarse.objective(uCoarse);

        % Now minimize the coarse model
        [uCoarseNew, ~, rhoCoarsePersist] = trustRegionStep(@fCoarse, uCoarse, Coarse.project(umin), Coarse.project(umax), rhoCoarse, rhoCoarsePersist, coarse_steps);
        % and prolongate the correction back to the fine space.
        uNew = u + Coarse.prolongate(uCoarseNew - uCoarse);
        uNew = max(umin, min(uNew, umax));

        % Check objective at trial point.
        [u_new, J_new, Ju_new, rhoCoarse, accepted, num_MC_sim_obj_est, num_MC_sim_grad_est] = acceptance(fine_fct, u, J, Ju, uNew, rhoCoarse, data);
        ABM_evaluations = ABM_evaluations + num_MC_sim_obj_est + 4 * num_MC_sim_grad_est * size(Ju_new, 2);

        % Update control, objective and gradient
        if accepted
            u = u_new;
            J = J_new;
            Ju = Ju_new;
        else
            writematrix(u_new', fullfile(data.data_path, data.output_directory, strcat("u_coarse_rejected_", int2str(itr), "_", int2str(counter_filename), ".csv")));
        end

        % Find latest Ju_aux_out_ file
        counter = 0;
        while isfile(fullfile(data.data_path, data.output_directory, strcat("Ju_aux_out_", int2str(counter), ".json")))
            counter = counter + 1;
        end
        % Undo last increment
        counter = counter - 1;
        
        % Load json
        fid = fopen(fullfile(data.data_path, data.output_directory, strcat("Ju_aux_out_", int2str(counter), ".json")), "r");
        raw = fread(fid,inf);
        str = char(raw');
        fclose(fid);
        Ju_aux_out_json = jsondecode(str); 
        
        % Append value and encode as json
        Ju_aux_out_json.accepted = accepted;
        json_encoded = jsonencode(Ju_aux_out_json);
        
        % Save json file
        fid = fopen(fullfile(data.data_path, data.output_directory, strcat("Ju_aux_out_", int2str(counter), ".json")), "w");
        fprintf(fid, json_encoded);
        fclose(fid);

        % Termination condition
        iter = iter + 1;
        if rhoCoarse < data.rho_min
            break;
        end
    end

    % Coarse first-order consistent model with linear correction term
    function [JC, JuC] = fCoarse(uC)
        [JC, JuC] = Coarse.objective(uC);
%         JC = JC + J - JCoarse + (JuCoarseRes(:) - JuCoarse(:))' * (uC(:) - uCoarse(:));
        JC = JC + (JuCoarseRes(:) - JuCoarse(:))' * (uC(:) - uCoarse(:));
        JuC = JuC + JuCoarseRes - JuCoarse;
    end

end

%% ------------------------------------------------------------------------

function [u, J, Ju, rho, accepted, num_MC_sim_obj_est, num_MC_sim_grad_est] = acceptance(fine_fct, u, J, Ju, uNew, rho, data)

    % Some settings
    estimate_objective = true;
    estimate_gradient = false;
    num_MC_sim_grad_est = 0;
    c1 = data.c1;

    % Compute delta_u and descent direction (vectors)
    delta_u = reshape(uNew - u, [], 1);
    s = -Ju(:);

    % Get eps accuracy of previous gradient approximation
    Eps = readmatrix(fullfile(data.data_path, "eps.csv"));

    % Sampling error bound
    error_e = Eps * c1 * dot(delta_u, s);
    fileID = fopen(fullfile(data.data_path, "error_e.csv"), "w");
    fprintf(fileID, num2str(error_e));
    
    % Output of current trust region size
    fprintf("Trust region radius rho = %e\n", rho)

    % No improvement can be expected
    if dot(delta_u, s) == 0
        fprintf("Reject fine step since no improvement can be expected due to <delta_u, s> = %g\n", dot(delta_u, s));

        % Decrease trust region radius
        rho = rho / 3;

        % Do not accept step
        accepted = false;
        num_MC_sim_obj_est = 0;
        num_MC_sim_grad_est = 0;
        return
    end

    % Calculate new objective value
    [JNew, ~, num_MC_sim_obj_est, ~, ~] = fine_fct(uNew, estimate_objective, estimate_gradient);

    % Check quality of model agreement
    if JNew - J <= -(1 + 3 * Eps) * c1 * dot(delta_u, s)
        fprintf("Accept fine step since %g <= %g\n", (JNew - J), -(1 + 3 * Eps) * c1 * dot(delta_u, s))

        % Increase trust region radius
        rho = 2 * rho;
        
        % Save trust region radius to file -> used for proper finite difference
        % step size choice
        fileID = fopen(strcat(data.data_path, "/rho.csv"), "w");
        fprintf(fileID, num2str(rho));

        % Some settings
        estimate_objective = false;
        estimate_gradient = true;

        % Compute new gradient
        [~, JuNew, ~, num_MC_sim_grad_est, ~] = fine_fct(uNew, estimate_objective, estimate_gradient);

        % Accept step
        u = uNew;
        J = JNew;
        Ju = JuNew;
        accepted = true;

    % No progress -> reject step and reduce trust region
    else
        % Decrease trust region radius
        rho = rho / 3;

        % Do not accept step
        accepted = false;
        fprintf("Reject fine step since %g !<= %g\n", (JNew - J), -(1 + 3 * Eps) * c1 * dot(delta_u, s));
    end
end

%% ------------------------------------------------------------------------

% Approximately minimizing an objective over a box using a trust region method
function [u, J, rho] = trustRegionStep(coarse_fct, u, umin, umax, rho, rho0, steps)
    umin = max(umin, u - rho);
    umax = min(umax, u + rho);

    [J, Ju] = coarse_fct(u);

    rho = rho0;
    for i=1:steps
        % Minimize (linear) Taylor model over trust region -> trial point
        [uNew, Jmod] = CauchyStep(J, Ju, u, umin, umax, rho);

        % Check objective at trial point.
        [JNew, ~] = coarse_fct(uNew);  % JuNew

        % Check quality of model agreement, reduce trust region if insufficient
        if JNew < J
            % Insufficient agreement -> reduce TR size
            if J - JNew < 0.5 * (J - Jmod)
                rho = rho / 2;
            % Excellent agreement -> increase TR size
            elseif J - JNew > 0.9 * (J - Jmod)
                rho = 2 * rho;
            end

            % Accept step
            u = uNew;
            J = JNew;
            break
        % No progress -> reject step and reduce trust region
        else
            rho = rho / 5;
        end
    end
end

%% ------------------------------------------------------------------------

% Minimizing a linear objective over an admissible box
% TODO: extend to quadratic objective, with SR1 quasi-Newton update
% J: objective value J(u)
% Ju: derivative
% u: current iterate
% umin, umax: box constraints
% rho: l-infinity trust region radius
% Output
% u: minimizer
% J: linear objective value at minimizer
function [u, J] = CauchyStep(J, Ju, u, umin, umax, rho)
    % u may be matrix-valued (components vertical, time steps horizontal). But this
    % doesn't play nicely with

    % In linear box-constrained optimization problems, the components are decoupled.
    % Thus, whenever the derivative Ju is negative, we take the upper bound,
    % if it's positive, we take the lower bound. The bounds are due to intersecting
    % the admissible region [umin,umax] with the trust region box [u-rho,u+rho].
    umin = max(umin, u - rho);
    umax = min(umax, u + rho);

    % project out active constraints
    Ju = Ju .* ((Ju<0) .* (umax-u>0) + (Ju>0).*(u-umin>0) + (Ju==0));

    % Now we need to compute the largest step in direction -Ju that is allowed,
    % i.e. leads to the boundary of the box. For that, we need to divide entrywise by
    % Ju. If there are entries 0, the result is completely NaN (even if it could be
    % +inf in our case), so we need to restrict the computation to the nonzero entries
    % in Ju.
    % Note that u may be matrix-valued (components vertical, time steps horizontal).
    % This doesn't play nicely with the indices from 'find', which are a row vector
    % for row vector Ju, but a column vector for matrix Ju. Here, we convert the
    % result to a column vector in any case.
    top = find(Ju < 0);
    bot = find(Ju > 0);

    if isempty([top(:); bot(:)])
    % projected gradient is apparently zero - we've found the solution already,
    % nothing to do.
        return
    end

    uminusumax = u - umax;
    uminusumin = u - umin;

    % find maximum step length that leads us to the boundary of the box
    alpha = min( [((uminusumax(top))./Ju(top)); ((uminusumin(bot))./Ju(bot))] );
    u = u - alpha * Ju;
    J = J - alpha * Ju(:)'*Ju(:);

    % Now, the update of u may lead to components which were supposed to become
    % active actually to be inactive due to rounding errors. This leads to
    % activity dection failure and consequently extremely small steps in the next
    % CauchyStep call. Thus, we enforce activity of components which are just a
    % tad off.
    top = find(abs(u - umax) < 1e-14 * norm([umax(:); u(:)], 'inf'));
    u(top) = umax(top);
    bot = find(abs(u - umin) < 1e-14 * norm([umin(:); u(:)], 'inf'));
    u(bot) = umin(bot);

end

%% ------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Computation of the reduced functional and its derivative by adjoint methods
% -------------------------------------------------------------------------

% Computes the reduced objective and its gradient, as well as state and adjoint trajectories
% y0 initial state value
% p  forward equation parameters
% u controls
% h  time step size
% forward: a function handle for the forward model rhs
% objective: a function handle for the objective
function [J, Ju, Y, L] = reducedObjective(y0, p, u, h, forward, objective, total_timesteps, duration_of_constant_control)
    U = stretch_vec(u(1, :), u(2, :), total_timesteps, duration_of_constant_control);
    [Y, J] = state(y0, p, U, h, forward, objective);
    [L, Ju_temp] = adjointState(Y, p, U, h, forward, objective);
    
    % Allocate memory
    Ju = zeros(size(u));
    
    % Integrate "pieceweise", i.e., per dimension wrt control
    for i=1:1:ceil(total_timesteps/duration_of_constant_control)
        Ju(:, i) = sum(Ju_temp(:, (i - 1) * duration_of_constant_control + 1:i*duration_of_constant_control), 2);
    end
end

%% ------------------------------------------------------------------------

% Integrate the primal trajectory and evaluate the objective
% y0: initial state value
% p:  parameter values
% us: a 2 x N array of control values
% h:  time step size
% forward: a function handle for the forward model rhs
% objective: a function handle for the objective
% Y: a n x N+1 array of state values
% J: the objective value
function [Y, J] = state(y0, p, u, h, forward, objective)
    J = 0;
    N = size(u, 2);
    Y = zeros(length(y0), N + 1);
    Y(:, 1) = y0;

    % Forward model (ODE model) - Explicit Euler
    for i=1:N
        Y(:, i + 1) = Y(:, i) + h * forward(Y(:, i), p, u(:, i));
        J = J + h * objective(Y(:, i), u(:, i));
    end
end

%% ------------------------------------------------------------------------

function [L, Ju] = adjointState(Y, p, u, h, forward, objective)
    N = size(u, 2);
    L = zeros(size(Y, 1), N + 1);

    Ju = zeros(size(u)); % to get the shape correct

    % Explicit Euler. This is the discrete adjoint.
    for j=N:-1:1
        L(:, j) = L(:, j + 1) - h * adjoint(L(:, j+1), Y(:, j), p, u(:, j), forward, objective);

        [~, ~, ju] = objective(Y(:, j), u(:, j));
        [~, ~, fu] = forward(Y(:, j), p, u(:, j));
        Ju(:, j) = h * (ju' + fu' * L(:, j+1));
    end
end

%% ------------------------------------------------------------------------

% Adjoint equation for multiplier (dual state) lambda. Note that this has to be
% integrated backwards in time.
function dl = adjoint(l, y, p, u, forward, objective)
    [~, Jy] = objective(y, u);
    [~, fy] = forward(y, p, u);
    dl = -(Jy' + fy' * l);
end

%% ------------------------------------------------------------------------

function [J, Ju, num_MC_sim_obj_est, num_MC_sim_grad_est, eps] = fine(u, estimate_objective, estimate_gradient, data)
    % 1) Save control in temp file
    writematrix(u', fullfile(data.data_path, "control_U.csv"));

    % Gradient estimate
    if estimate_gradient
        fprintf("\nGradient estimate...\n")
        system("chmod +x GERDA_gradient_estimation_dummy.py; ./GERDA_gradient_estimation_dummy.py");
    end

    % Estimate J
    if estimate_objective
        fprintf("\nObjective estimate...\n")
        system("chmod +x GERDA_state_estimation_dummy.py; ./GERDA_state_estimation_dummy.py");
    end

    % Read J, Ju, samplesizes and estimate gradient eps accuracy
    J = readmatrix(fullfile(data.data_path, "objective_estimate.csv"));
    Ju = readmatrix(fullfile(data.data_path, "gradient_estimate_projected.csv"))';
    num_MC_sim_obj_est = readmatrix(fullfile(data.data_path, "num_MC_sim_obj_est.csv"));
    num_MC_sim_grad_est = readmatrix(fullfile(data.data_path, "num_MC_sim_grad_est.csv"));
    eps = readmatrix(fullfile(data.data_path, "eps.csv"));

end

%% ------------------------------------------------------------------------

function u = stretch_vec(Us, Uw, total_timesteps, duration_of_constant_control)
    u = zeros(2, total_timesteps);

    sim_time = 0;
    week_num = 0;

    while sim_time < total_timesteps
        % Complete weeks
        if (total_timesteps - week_num * duration_of_constant_control) >= duration_of_constant_control
            u(1, 1 + week_num * duration_of_constant_control: (week_num + 1) * duration_of_constant_control) = Us(week_num + 1);
            u(2, 1 + week_num * duration_of_constant_control: (week_num + 1) * duration_of_constant_control) = Uw(week_num + 1);
            sim_time = sim_time + duration_of_constant_control;
        % Incomplete week
        else
            u(1, 1 + week_num * duration_of_constant_control:end) = Us(week_num + 1);
            u(2, 1 + week_num * duration_of_constant_control:end) = Uw(week_num + 1);
            sim_time = sim_time + total_timesteps - week_num * duration_of_constant_control;
        end
        week_num = week_num + 1;
    end
end

%% ------------------------------------------------------------------------

function U = condense_vec(u, duration_of_constant_control)
    U = u(:, 1:duration_of_constant_control:end);
end

%% ODE---------------------------------------------------------------------

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
         Ic_Rc ];

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

%% ------------------------------------------------------------------------

% Objective for control of the SIR model. This gives the Lagrange term to be
% integrated over time.
function [J, Jy, Ju] = objective(y, u)
    Ia = y(3); Ic = y(4);

    n_people = sum(y);
    so = 1;  % Weights for social impact
    ho = 1;  % Weights for economical impact
    imax = 0.005;
    u2max = 0.8 + 0.1;

    if u(2) >= u2max
        keyboard
        J = inf;
        Jy = [0 0 1 1 0 0];
        Ju = [1 0];
    else
        J = (Ia + Ic) / n_people + exp(10 * ((Ia + Ic) - imax * n_people) / n_people) + so * u(1)^2 - ho * log(u2max - u(2));
        Jy = [0, 0, (1 + 10 * exp(10 / n_people * (Ia + Ic - imax * n_people))) / n_people, (1 + 10 * exp(10 / n_people * (Ia + Ic - imax * n_people))) / n_people, 0, 0];
        Ju = [2 * so * u(1), ho / (u2max - u(2))];
    end
end

%% ------------------------------------------------------------------------

function [y] = WeeklyConst(x, duration_of_constant_control)

    num_weeks = floor(size(x, 2) / duration_of_constant_control);
    T = length(x);
    sum_x = zeros(2, num_weeks);

    for i=1:num_weeks
        sum_x(:, i) = sum(x(:, (i - 1) * duration_of_constant_control + 1 : i * duration_of_constant_control), 2);
    end

    x1 = sum_x./duration_of_constant_control;

    if num_weeks * duration_of_constant_control < T
        x1 = [x1, sum(x(:, num_weeks * duration_of_constant_control + 1 : end), 2) ./ (T - num_weeks * duration_of_constant_control)];
    end

    y = [];
    for i=1:num_weeks
        y = [y, x1(:, i) * ones(1, duration_of_constant_control)];
    end

    if num_weeks * duration_of_constant_control < T
        y = [y, x1(:, end) * ones(1, T - num_weeks * duration_of_constant_control)];
    end
end

%% ------------------------------------------------------------------------

function [du] = gradproject(du, u)
    for i=1:length(u)
        if (u(i) == 0) && (du(i) < 0) || (u(i) == 1) && (du(i) > 0)
            du(i) = 0;
        end
    end
end

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

%% Fit ODE

% Full observability
function [p_full] = ODE_fit_full(y0, u, h, YmeanABM, YvarABM)

    YvarABM(YvarABM==0) = 1e-9;
    r = optimvar('r', 5, "LowerBound", 0, "UpperBound", 0.1);

    optifun = fcn2optimexpr(@integrate_ODE, y0, r, u, h);
    obj = sum(sum(((optifun - YmeanABM).^2)./YvarABM));

    prob = optimproblem('ObjectiveSense', 'minimize', 'Objective', obj);

    % Settings, 3000 is default
    options = optimoptions(prob, 'MaxFunctionEvaluations', 10000, 'Display', 'off');

    % Initial value for nonlinear problems, magnitudes already correct
    r0.r = [1e-6, 1e-4, 1e-4, 1e-2, 1e-2];

    [rsol, ~] = solve(prob, r0, 'Options', options);
    p_full = rsol.r;

end

% Partial observability
function [p_full] = ODE_fit_partial(y0, u, h, YmeanABM, YvarABM)

    YvarABM(YvarABM==0) = 1e-9;
    r = optimvar('r', 5, "LowerBound", 0, "UpperBound", 0.1);

    optifun = fcn2optimexpr(@integrate_ODE_partial, y0, r, u, h);
    obj = sum(sum(((optifun - YmeanABM).^2)./YvarABM));

    prob = optimproblem('ObjectiveSense', 'minimize', 'Objective', obj);

    % Settings, 3000 is default
    options = optimoptions(prob, 'MaxFunctionEvaluations', 10000, 'Display', 'off');

    % Initial value for nonlinear problems, magnitudes already correct
    r0.r = [1e-6, 1e-4, 1e-4, 1e-2, 1e-2];

    [rsol, ~] = solve(prob, r0, 'Options', options);
    p_full = rsol.r;

end
