%% Tutorial 1: Running Uncertainty Quantification (UQ) with mfoil and Monte Carlo Standard Error
%
% This script demonstrates how to:
% - Generate Monte Carlo samples for aerodynamic analysis using mfoil (MATLAB version).
% - Evaluate the solver with generated samples.
% - Quantify uncertainty in the outputs using Monte Carlo standard error (SE).
%
% Steps:
% 1. Define input parameter ranges for uniform sampling.
% 2. Generate sample input CSV file.
% 3. Run the mfoil solver on the samples.
% 4. Compute SE for mean, standard deviation, and P-box probability using UQ analysis.
%
% Note: MATLAB version automatically uses mfoil solver regardless of CSV header.

clear; clc; close all;

% Add the path to your MATLAB classes (adjust as needed)
addpath('src');

% Add mfoil to the path (adjust path as needed)
addpath('/Users/aidanlay/UQ_Research/UQ_mxfoil_MATLAB/src/solvers');

% NOTE:
% MATLAB version uses mfoil solver which supports forced transition (xtr_upper, xtr_lower).
% Python version uses xfoil solver with forced transition support.

% Define the input parameters for the uniform distribution
input_names = {'alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower'};

% Specify lower and upper bounds for each input parameter
xmin = [-0.3, 492500, -0.24, 0.255, 0.637];
xmax = [0.3, 507500, 0.24, 0.345, 0.763];

% Step 1: Create a sample input file with uniform distribution
% MATLAB version automatically uses mfoil solver
fprintf('Step 1: Creating uniform samples...\n');
sampler = sample(100, 'mfoil');
sampler.create_samples('uniform_samples.csv', 'uniform', 'monte', xmin, xmax, ...
                      'input_names', input_names);

% Step 2: Evaluate the solver with the generated samples
fprintf('Step 2: Evaluating solver with generated samples...\n');
solver = solver_eval('uniform_samples.csv');
solver.run();

% Step 3: Quantify the sample input uncertainty using Monte Carlo SE
fprintf('Step 3: Computing Monte Carlo Standard Error...\n');
uqa = uq_analysis();
SE = uqa.monte_carlo_SE('uniform_samples.csv');

% Step 4: Print the SE results for mean, standard deviation, and P-box probability
fprintf('\n=== Results ===\n');
fprintf('SE Mean: [%.6f, %.6f]\n', SE{1}(1), SE{1}(2));                 % Uncertainty interval for mean of Cl and Cm
fprintf('SE Standard Deviation: [%.6f, %.6f]\n', SE{2}(1), SE{2}(2));   % Uncertainty interval for stddev of Cl and Cm
fprintf('SE P-box Probability: %.6f\n', SE{3});                         % Uncertainty interval for probability outside P-box

fprintf('\nTutorial 1 completed successfully!\n');
