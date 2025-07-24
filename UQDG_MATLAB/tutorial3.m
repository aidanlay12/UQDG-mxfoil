%% Tutorial 3: Kriging Surrogate Modeling for Aerodynamic Analysis
%
% This script demonstrates how to:
% - Generate training and validation samples for mfoil.
% - Evaluate the solver on both sets.
% - Assemble a kriging surrogate model using training data.
% - Evaluate the surrogate on validation data.
% - Compute RMSE between surrogate predictions and actual solver outputs.
%
% Steps:
% 1. Define input parameter ranges for uniform sampling.
% 2. Generate training samples for kriging.
% 3. Run the solver on training samples.
% 4. Generate validation samples (e.g., using Monte Carlo).
%    NOTE: If 'validation_samples.csv' already exists, comment out the creation line below to avoid appending to previous files.
% 5. Run the solver on validation samples.
% 6. Assemble and save kriging surrogate model.
% 7. Evaluate surrogate predictions on validation samples.
% 8. Compute and print RMSE for surrogate accuracy.
%
% NOTE: MATLAB version automatically uses mfoil solver.

clear; clc; close all;

% Add the path to your MATLAB classes (adjust as needed)
addpath('src');

% Add mfoil to the path (adjust path as needed)
addpath('/Users/aidanlay/UQ_Research/UQ_mxfoil_MATLAB/src/solvers');

% Define the input parameters for the uniform distribution
input_names = {'alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower'};

% Specify lower and upper bounds for each input parameter
xmin = [-0.3, 492500, -0.24, 0.255, 0.637];
xmax = [0.3, 507500, 0.24, 0.345, 0.763];

% % Step 1: Create a training input file for kriging
% % NOTE: MATLAB version automatically uses mfoil solver
% fprintf('Step 1: Creating mixed kriging training samples...\n');
% training_sampler = sample(50, 'mfoil');
% training_sampler.mix_krig('training_samples_krig.csv', xmin, xmax, ...
%                          'input_names', input_names);

% % Step 2: Evaluate the solver with the training samples
% fprintf('Step 2: Evaluating solver with training samples...\n');
% training_solver = solver_eval('training_samples_krig.csv');
% training_solver.run();

% % Step 3: Create a validation input file using Monte Carlo sampling
% % WARNING: If 'validation_samples.csv' already exists, comment out the line below to avoid appending to previous files.
% fprintf('Step 3: Creating validation samples using Monte Carlo...\n');
% validation_sampler = sample(100, 'mfoil');
% validation_sampler.create_samples('validation_samples.csv', 'uniform', 'monte', ...
%                                  xmin, xmax, 'input_names', input_names);

% % Step 4: Evaluate the solver with the validation samples
% fprintf('Step 4: Evaluating solver with validation samples...\n');
% validation_solver = solver_eval('validation_samples.csv');
% validation_solver.run();

% Step 5: Assemble the kriging surrogate model using training data
fprintf('Step 5: Assembling kriging surrogate model...\n');
krig = krig_model();
krig.assemble_surrogate('training_samples_krig.csv', ...    % Training data file
                       'krig_model.txt', ...                % Surrogate model file
                       xmin, ...                             % Minimum values for normalization
                       xmax);                                % Maximum values for normalization

% Step 6: Evaluate the surrogate model on validation samples
fprintf('Step 6: Evaluating kriging surrogate on validation samples...\n');
krig.evaluate_surrogate('validation_samples.csv', ...       % CSV file with input samples
                       'krig_model.txt');                   % Surrogate model file

% Step 7: Compute RMSE between surrogate predictions and actual outputs
fprintf('Step 7: Computing RMSE between surrogate and actual outputs...\n');
uqa = uq_analysis();
RMSE = uqa.surrogate_RMSE('validation_samples_out.csv', ...
                         'validation_samples_krig_model_sout.csv');

% Step 8: Print RMSE for Cl and Cm
fprintf('\n=== Results ===\n');
fprintf('Cl surrogate RMSE: %.6f\n', RMSE(1));
fprintf('Cm surrogate RMSE: %.6f\n', RMSE(2));

fprintf('\nTutorial 3 completed successfully!\n');
