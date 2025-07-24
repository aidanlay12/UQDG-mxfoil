%% Tutorial 2: Polynomial Chaos Expansion Surrogate Modeling for Aerodynamic Analysis
%
% This script demonstrates how to:
% - Generate training and validation samples for mfoil (MATLAB version).
% - Evaluate the solver on both sets.
% - Assemble a polynomial chaos surrogate model using training data.
% - Evaluate the surrogate on validation data.
% - Compute RMSE between surrogate predictions and actual solver outputs.
%
% Steps:
% 1. Define input parameter ranges for uniform sampling.
% 2. Generate training samples (e.g., using Sobol sequence).
% 3. Run the mfoil solver on training samples.
% 4. Generate validation samples (e.g., using Monte Carlo).
%    NOTE: If 'validation_samples.csv' already exists, comment out the creation line below to avoid appending to previous files.
% 5. Run the mfoil solver on validation samples.
% 6. Assemble and save polynomial chaos surrogate coefficients.
% 7. Evaluate surrogate predictions on validation samples.
% 8. Compute and print RMSE for surrogate accuracy.
%
% Note: MATLAB version automatically uses mfoil solver regardless of CSV header.

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

% Step 1: Create a training input file using Sobol sequence
% MATLAB version automatically uses mfoil solver
fprintf('Step 1: Creating training samples using Sobol sequence...\n');
training_sampler = sample(50, 'mfoil');
training_sampler.create_samples('training_samples_poly.csv', 'uniform', 'sobol', ...
                               xmin, xmax, 'input_names', input_names);

% Step 2: Evaluate the solver with the training samples
fprintf('Step 2: Evaluating solver with training samples...\n');
training_solver = solver_eval('training_samples_poly.csv');
training_solver.run();

% Step 3: Create a validation input file using Monte Carlo sampling
% WARNING: If 'validation_samples.csv' already exists, comment out the line below to avoid appending to previous files.
fprintf('Step 3: Creating validation samples using Monte Carlo...\n');
validation_sampler = sample(100, 'mfoil');
validation_sampler.create_samples('validation_samples.csv', 'uniform', 'monte', ...
                                 xmin, xmax, 'input_names', input_names);

% Step 4: Evaluate the solver with the validation samples
fprintf('Step 4: Evaluating solver with validation samples...\n');
validation_solver = solver_eval('validation_samples.csv');
validation_solver.run();

% Step 5: Assemble the polynomial chaos surrogate model using training data
fprintf('Step 5: Assembling polynomial chaos surrogate model...\n');
poly = poly_model();
poly.assemble_surrogate('training_samples_poly.csv', ...    % CSV file with input samples
                       2, ...                               % Degree of polynomial chaos
                       xmin, ...                            % Minimum values for normalization
                       xmax, ...                            % Maximum values for normalization
                       'poly_coeffs.csv');                  % Output CSV for surrogate coefficients

% Step 6: Evaluate the surrogate model on validation samples
fprintf('Step 6: Evaluating surrogate model on validation samples...\n');
poly.evaluate_surrogate('validation_samples.csv', ...       % CSV file with validation input samples
                       'poly_coeffs.csv');                  % CSV file with saved coefficients

% Step 7: Compute RMSE between surrogate predictions and actual outputs
fprintf('Step 7: Computing RMSE between surrogate and actual outputs...\n');
uqa = uq_analysis();
RMSE = uqa.surrogate_RMSE('validation_samples_out.csv', ...
                         'validation_samples_poly_coeffs_sout.csv');

% Step 8: Print RMSE for Cl and Cm
fprintf('\n=== Results ===\n');
fprintf('Cl surrogate RMSE: %.6f\n', RMSE(1));
fprintf('Cm surrogate RMSE: %.6f\n', RMSE(2));

fprintf('\nTutorial 2 completed successfully!\n');
