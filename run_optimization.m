% clc;
% clear;
% close all;

%% optimize
% optimize the initial conditions and controller hyper parameters
q0 = [pi/9; -pi/9; 0];
dq0 = [0; 0; 8]; 
X0 = [q0; dq0; control_hyper_parameters()];

% use fminsearch and optimset to control the MaxIter
options = optimset('MaxIter', 1000);
X_opt = fminsearch(@optimization_fun, X0, options);

%% simulate solution

% extract parameters
q0_opt = X_opt(1:3);
dq0_opt = X_opt(4:6);
x_opt = X_opt(7:end);

% simulate
num_steps = 20;
sln = solve_eqns(q0_opt, dq0_opt, num_steps, x_opt);
animate(sln);
results = analyse(sln, x_opt, true);
