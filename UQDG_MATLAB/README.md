# UQDG-mxfoil MATLAB Implementation

This directory contains the MATLAB implementation of UQDG-mxfoil, an educational uncertainty quantification library for aerodynamic analysis using the mfoil solver.

## Quick Start

### Installation

1. **Navigate to the MATLAB directory:**
   ```bash
   cd UQDG_MATLAB/
   ```

2. **Add source directory to MATLAB path:**
   ```matlab
   addpath('src');
   addpath('src/solvers');
   ```

3. **Run a tutorial to verify installation:**
   ```matlab
   tutorial1;  % Basic Monte Carlo UQ
   ```

### Requirements

- **MATLAB**: R2019b or later
- **Toolboxes**: Statistics and Machine Learning Toolbox (recommended)
- **External Solvers**: 
  - mfoil toolbox (for multi-element airfoil analysis)
- **Optional**: Optimization Toolbox (for Kriging hyperparameter fitting)

### Basic Usage

```matlab
% Add paths
addpath('src');

% Generate samples
sample_obj = sample(100, 'mfoil');
sample_obj.create_samples('test_samples.csv', 'uniform', 'monte', [-0.3, 492500], [0.3, 507500]);

% Run simulations
solver = solver_eval('test_samples.csv');
solver.run();

% Analyze uncertainty
uq_obj = uq_analysis();
results = uq_obj.monte_carlo_analysis('test_samples.csv');
```

## Directory Structure

```
UQDG_MATLAB/
├── README.md                    # This file
├── tutorial1.m                  # Monte Carlo UQ tutorial
├── tutorial2.m                  # Polynomial Chaos tutorial
├── tutorial3.m                  # Kriging surrogate tutorial
├── input/                       # Input files directory
│   ├── README.md
│   ├── *.csv                    # Sample input files
│   ├── *.in                     # Configuration files
│   └── *.sur                    # Surrogate model files
├── output/                      # Output files directory
│   ├── README.md
│   ├── *_out.csv               # Solver results
│   ├── *_fail.csv              # Failed simulation cases
│   └── *_sout.csv              # Surrogate model predictions
└── src/                         # Source code
    ├── sample.m                 # Sampling module
    ├── solver_eval.m            # Dual solver interface
    ├── uq_analysis.m            # UQ analysis tools
    ├── poly_model.m             # Polynomial Chaos modeling
    ├── krig_model.m             # Kriging surrogate modeling
    └── solvers/                 # External solvers
        └── mfoil.m              # mfoil interface
```

## Core Functions

### `sample.m`
- **Purpose**: Generate samples for uncertainty quantification
- **Distributions**: Uniform, normal, beta, triangular, lognormal
- **Sampling methods**: Monte Carlo, Latin Hypercube, Sobol sequences
- **Special features**: GCI sample generation, multi-variable support

### `solver_eval.m`
- **Purpose**: Interface to mfoil aerodynamic solver for multi-element airfoil analysis
- **Features**: Batch processing, error handling, progress tracking
- **Input format**: Double-header CSV files
- **Outputs**: Lift coefficient (Cl), moment coefficient (Cm)

### `uq_analysis.m`
- **Purpose**: Uncertainty quantification and statistical analysis
- **Capabilities**: Statistical moments, Monte Carlo analysis, failure rate calculation
- **Metrics**: Mean, variance, skewness, kurtosis, confidence intervals

### `poly_model.m`
- **Purpose**: Polynomial Chaos Expansion surrogate modeling
- **Features**: Coefficient calculation, model evaluation, validation metrics
- **Applications**: Fast uncertainty propagation, sensitivity analysis

### `krig_model.m`
- **Purpose**: Kriging surrogate modeling
- **Features**: Gaussian process regression, variogram fitting, prediction uncertainty
- **Applications**: Non-linear response surface modeling, spatial interpolation

## Key Features

### mfoil Solver Integration
- **Multi-element airfoils**: Complete support for multi-element airfoil configurations
- **Flap deflection**: Advanced flap deflection and control surface modeling
- **NACA airfoils**: Built-in NACA airfoil generation and modification
- **Robust integration**: Complete parameter passing and result extraction

### Advanced Sampling
- **Multiple methods**: Monte Carlo, Latin Hypercube, Sobol quasi-random
- **Distribution flexibility**: Support for various probability distributions
- **Efficient space-filling**: Optimized sampling for high-dimensional problems
- **Mesh convergence studies**: Specialized GCI analysis capabilities

### Comprehensive UQ Analysis
- **Statistical moments**: Complete statistical characterization
- **Failure analysis**: Automatic detection and reporting of failed simulations
- **Uncertainty bounds**: Confidence intervals and prediction intervals
- **Visualization**: Built-in plotting capabilities for results

## Input/Output Format

### CSV Input Format (Double Header)
```csv
solver,num_variables,num_samples
alpha,Re,flap_deflection,xtr_upper,xtr_lower
-0.1,500000,0.1,0.3,0.7
0.0,500000,0.0,0.3,0.7
0.1,500000,-0.1,0.3,0.7
```

### Solver Selection
- **`mfoil`**: Use mfoil solver for multi-element airfoil configurations

### Output Files
- **`*_out.csv`**: Solver results (Cl, Cm columns)
- **`*_fail.csv`**: Failed simulation cases with input parameters
- **`*_sout.csv`**: Surrogate model predictions and uncertainties

## Tutorials

The implementation includes three comprehensive tutorials:

1. **`tutorial1.m`**: Basic Monte Carlo uncertainty quantification
   - Uniform and normal distribution sampling
   - mfoil solver execution
   - Statistical moment calculation and visualization

2. **`tutorial2.m`**: Polynomial Chaos surrogate modeling
   - Training data generation with structured sampling
   - Polynomial coefficient calculation and validation
   - Surrogate accuracy assessment with RMSE metrics

3. **`tutorial3.m`**: Kriging surrogate modeling
   - Kriging model training with hyperparameter optimization
   - Prediction on validation datasets
   - Uncertainty quantification with prediction intervals

## Running Tutorials

Execute tutorials from the MATLAB command window:

```matlab
% Ensure paths are set
addpath('src');

% Run tutorials
tutorial1;    % Basic UQ workflow
tutorial2;    % Polynomial Chaos modeling
tutorial3;    % Kriging surrogate modeling
```

## Advanced Usage

### Custom Solver Parameters
```matlab
% Custom solver configuration
solver = solver_eval('input.csv');
solver.panel_size = 512;
solver.convergence_tolerance = 1e-6;
solver.max_iterations = 1000;
solver.run();
```

### Batch Processing
```matlab
% Process multiple input files
input_files = {'case1.csv', 'case2.csv', 'case3.csv'};
for i = 1:length(input_files)
    solver = solver_eval(input_files{i});
    solver.run();
end
```

### Custom Sampling Strategies
```matlab
% Beta distribution with Sobol sampling
sample_obj = sample(500, 'mfoil');
sample_obj.create_samples('beta_samples.csv', 'beta', 'sobol', ...
    [2, 3], [5, 7], {'alpha1', 'alpha2'});
```

### Surrogate Model Workflows
```matlab
% Complete surrogate modeling workflow
% 1. Generate training data
sample_obj = sample(200, 'mfoil');
sample_obj.create_samples('training.csv', 'uniform', 'sobol', xmin, xmax);

% 2. Run solver on training data
solver = solver_eval('training.csv');
solver.run();

% 3. Train surrogate model
poly_obj = poly_model();
poly_obj.assemble_poly('training.csv');

% 4. Generate validation data and evaluate surrogate
sample_obj.create_samples('validation.csv', 'uniform', 'monte', xmin, xmax);
poly_obj.eval_poly('validation.csv');
```

## Troubleshooting

### Common Issues

1. **Path errors**: Ensure source directory is in MATLAB path
   ```matlab
   addpath(genpath('src'));
   ```

2. **mfoil toolbox missing**: Install mfoil toolbox for multi-element analysis
   ```matlab
   % Check if mfoil is available
   which mfoil
   ```

3. **Memory issues with large datasets**: Use batch processing
   ```matlab
   % Process data in chunks
   solver.batch_size = 50;  % Process 50 cases at a time
   ```

### Performance Tips

- Use Sobol sequences for better space-filling properties in high dimensions
- Increase panel size for higher accuracy (trades off with computational time)
- Monitor failed simulations in `*_fail.csv` files
- Use surrogate models for large-scale uncertainty studies (>1000 samples)
- Enable parallel processing for batch simulations if Parallel Computing Toolbox is available

## Dependencies

### Required
- **MATLAB**: R2019b or later for full functionality
- **Base MATLAB**: Core functionality without additional toolboxes

### Recommended
- **Statistics and Machine Learning Toolbox**: Advanced sampling methods, distribution fitting
- **Optimization Toolbox**: Kriging hyperparameter optimization
- **Parallel Computing Toolbox**: Batch processing acceleration

### External Solvers
- **mfoil toolbox**: Multi-element airfoil analysis (separate installation)

## Solver-Specific Features

### mfoil Integration
- NACA airfoil generation and modification
- Flap deflection and multi-element configurations
- Operating condition setup (angle of attack, Reynolds number)
- Complete integration with uncertainty quantification workflows

## Support

For questions or issues:
- Check tutorial examples for usage patterns
- Review input/output file formats in the documentation
- Examine solver convergence parameters and adjust as needed
- Consult the main repository README for additional context
- Verify MATLAB toolbox availability using `ver` command
