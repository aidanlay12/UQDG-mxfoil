# UQDG-mxfoil MATLAB Conversion

This directory contains the MATLAB conversion of the UQDG-mxfoil Python project for uncertainty quantification and aerodynamic analysis using xfoil and mfoil solvers.

## Converted Files

### Core Classes (src/)
- **sample.m** - Unified sampler for various probability distributions (Monte Carlo, Latin Hypercube, Sobol)
- **solver_eval.m** - Aerodynamic analysis solver supporting both xfoil and mfoil with complete mfoil integration
- **uq_analysis.m** - Uncertainty quantification analysis with statistical moment calculations
- **krig_model.m** - Kriging surrogate model implementation
- **poly_model.m** - Polynomial Chaos surrogate model implementation

### Tutorial Scripts
- **tutorial1.m** - Monte Carlo sampling and uncertainty quantification
- **tutorial2.m** - Polynomial surrogate modeling
- **tutorial3.m** - Kriging surrogate modeling  

### Directories
- **input/** - Input CSV files for simulations
- **output/** - Results and output files

## Key Features

### solver_eval.m Capabilities
- **Dual solver support**: Both xfoil and mfoil solvers fully implemented
- **mfoil integration**: Complete mfoil solver with NACA airfoil support, flap deflection, and operating condition setup
- **Error handling**: Robust error handling for failed simulations
- **CSV I/O**: Double-header CSV input format with automatic result output
- **Progress tracking**: Real-time progress reporting with time estimates

### sample.m Features
- **Multiple sampling methods**: Monte Carlo, Latin Hypercube, Sobol sequences
- **Distribution support**: Normal, uniform, triangular, lognormal distributions
- **Mesh convergence**: GCI analysis support with structured mesh refinement

### UQ Analysis
- **Statistical moments**: Mean, variance, skewness, kurtosis calculations
- **Failure rate analysis**: Automatic detection and reporting of simulation failures
- **Surrogate modeling**: Both Kriging and Polynomial Chaos implementations

## Usage

```matlab
% Add source directory to path
addpath('src');

% Run a tutorial
tutorial1;  % Monte Carlo UQ analysis

% Or create custom analysis
solver = solver_eval('input_file.csv');
solver.run();
```

## Input File Format

CSV files should have a double header:
```
solver_type,num_variables,num_samples
variable_name1,variable_name2,...
data_row1_values
data_row2_values
...
```

Example:
```
mfoil,5,100
alpha,Re,flap_deflection,xtr_upper,xtr_lower
2.5,100000,0.0,1.0,1.0
3.0,150000,5.0,0.9,0.8
...
```

## Dependencies

- MATLAB R2019b or later
- mfoil toolbox (for mfoil solver functionality)
- Statistics and Machine Learning Toolbox (for advanced sampling methods)
- Optimization toolbox for Kriging variogram fitting
