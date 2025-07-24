# UQDG-mxfoil Python Implementation

This directory contains the Python implementation of UQDG-mxfoil, an educational uncertainty quantification library for aerodynamic analysis using XFOIL solver.

## Quick Start

### Installation

1. **Navigate to the Python directory:**
   ```bash
   cd UQDG_Python/
   ```

2. **Install the package:**
   ```bash
   pip install .
   ```

   For development installation (editable):
   ```bash
   pip install -e .
   ```

3. **Install cfd-verify (required for GCI analysis):**
   ```bash
   git clone https://github.com/ORNL/cfd-verify.git
   cd cfd-verify
   pip install -e .
   cd ..
   ```

### Requirements

- **Python**: ≥3.9
- **Dependencies**: numpy, scipy, matplotlib, pandas
- **External Solver**: XFOIL (included in `src/solvers/`)
- **GCI Analysis**: cfd-verify package (for Tutorial 4)

### Basic Usage

```python
import UQDGmxfoil.sample as smp
import UQDGmxfoil.solver_eval as xmeval
import UQDGmxfoil.uq_analysis as uq

# Generate samples
smp.sample(num_samples=100, solver='xfoil').create_samples(
    'test_samples.csv', 'uniform', 'monte', [-0.3, 492500], [0.3, 507500])

# Run simulations
xmeval.solver_eval('test_samples.csv').run()

# Analyze uncertainty
SE = uq.uq_analysis().monte_carlo_SE('test_samples.csv')
```

## Directory Structure

```
UQDG_Python/
├── README.md                    # This file
├── pyproject.toml              # Package configuration
├── tutorial1.py                # Monte Carlo UQ tutorial
├── tutorial2.py                # Polynomial Chaos tutorial
├── tutorial3.py                # Kriging surrogate tutorial
├── tutorial4.py                # Grid Convergence Index tutorial
├── input/                      # Input files directory
│   ├── README.md
│   └── *.csv                   # Sample input files
├── output/                     # Output files directory
│   ├── README.md
│   └── *_out.csv              # Results from simulations
└── src/                        # Source code
    ├── UQDGmxfoil/            # Main Python package
    │   ├── README.md
    │   ├── sample.py          # Sampling module
    │   ├── solver_eval.py     # Solver interface
    │   ├── uq_analysis.py     # UQ analysis tools
    │   ├── poly_model.py      # Polynomial Chaos
    │   └── krig_model.py      # Kriging surrogate
    ├── solvers/               # External solvers
    │   └── xfoil              # XFOIL executable
    └── UQDGmxfoil.egg-info/   # Package metadata
```

## Core Modules

### `sample.py`
- **Purpose**: Generate samples for uncertainty quantification
- **Distributions**: Uniform, normal, beta, triangular
- **Sampling methods**: Monte Carlo, Latin Hypercube, Sobol sequences
- **Special features**: GCI sample generation for mesh convergence studies

### `solver_eval.py`
- **Purpose**: Interface to XFOIL aerodynamic solver
- **Features**: Batch processing, error handling, progress tracking
- **Input format**: Double-header CSV files
- **Outputs**: Lift coefficient (Cl), moment coefficient (Cm)

### `uq_analysis.py`
- **Purpose**: Uncertainty quantification and statistical analysis
- **Capabilities**: Monte Carlo standard error, GCI analysis
- **Metrics**: Mean, standard deviation, confidence intervals

### `poly_model.py`
- **Purpose**: Polynomial Chaos Expansion surrogate modeling
- **Features**: Coefficient calculation, model evaluation, RMSE assessment
- **Applications**: Fast uncertainty propagation, sensitivity analysis

### `krig_model.py`
- **Purpose**: Kriging surrogate modeling
- **Features**: Gaussian process regression, hyperparameter optimization
- **Applications**: Non-linear response surface modeling

## Input/Output Format

### CSV Input Format (Double Header)
```csv
solver,num_variables,num_samples
alpha,Re,flap_deflection,xtr_upper,xtr_lower
-0.1,500000,0.1,0.3,0.7
0.0,500000,0.0,0.3,0.7
0.1,500000,-0.1,0.3,0.7
```

### Output Files
- **`*_out.csv`**: Solver results (Cl, Cm columns)
- **`*_fail.csv`**: Failed simulation cases
- **`*_sout.csv`**: Surrogate model predictions

## Tutorials

The implementation includes four comprehensive tutorials:

1. **`tutorial1.py`**: Basic Monte Carlo uncertainty quantification
   - Uniform sampling generation
   - XFOIL simulation execution
   - Monte Carlo standard error calculation

2. **`tutorial2.py`**: Polynomial Chaos surrogate modeling
   - Training data generation with Sobol sequences
   - Polynomial coefficient calculation
   - Validation and RMSE assessment

3. **`tutorial3.py`**: Kriging surrogate modeling
   - Kriging model training
   - Prediction on validation data
   - Accuracy evaluation

4. **`tutorial4.py`**: Grid Convergence Index (GCI) analysis
   - Mesh refinement study setup
   - Numerical uncertainty quantification
   - Convergence analysis

## Running Tutorials

Execute tutorials from the UQDG_Python directory:

```bash
python tutorial1.py    # Basic UQ workflow
python tutorial2.py    # Polynomial Chaos
python tutorial3.py    # Kriging modeling
python tutorial4.py    # GCI analysis
```

## Advanced Usage

### Custom Solver Parameters
```python
# Custom panel size and convergence criteria
solver = xmeval.solver_eval('input.csv', panel_size=512, conv_tol=1e-6)
solver.run()
```

### Single Case Evaluation
```python
# Run specific sample index
solver.single_solve(case_index=0, include_eps=True)
```

### Custom Sampling
```python
# Beta distribution sampling
smp.sample(100, 'xfoil').create_samples(
    'beta_samples.csv', 'beta', 'sobol', 
    xmin=[2, 3], xmax=[5, 7], input_names=['alpha1', 'alpha2'])
```

## Troubleshooting

### Common Issues

1. **XFOIL not found**: Ensure XFOIL executable has proper permissions
   ```bash
   chmod +x src/solvers/xfoil
   ```

2. **Convergence failures**: Adjust solver parameters
   ```python
   solver = xmeval.solver_eval('input.csv', num_of_iter=1000, conv_tol=1e-5)
   ```

3. **Import errors**: Ensure package is properly installed
   ```bash
   pip install -e .
   ```

4. **GCI analysis errors**: Install cfd-verify package
   ```bash
   git clone https://github.com/ORNL/cfd-verify.git
   cd cfd-verify
   pip install -e .
   cd ..
   ```

### Performance Tips

- Use Sobol sequences for better space-filling properties
- Increase panel size for higher accuracy (at computational cost)
- Monitor failed simulations in `*_fail.csv` files
- Use surrogate models for large-scale UQ studies

## Dependencies

The package automatically installs required dependencies:
- `numpy`: Numerical computations
- `scipy`: Scientific computing and optimization
- `matplotlib`: Plotting and visualization
- `pandas`: Data manipulation and I/O

### Additional Dependencies

For Grid Convergence Index (GCI) analysis in Tutorial 4:
- `cfd-verify`: ORNL's verification and validation toolkit
  - Install: Clone repository and install in development mode
    ```bash
    git clone https://github.com/ORNL/cfd-verify.git
    cd cfd-verify
    pip install -e .
    ```
  - Repository: https://github.com/ORNL/cfd-verify

## Support

For questions or issues:
- Check tutorial examples for usage patterns
- Review input/output file formats
- Examine solver convergence parameters
- Consult the main repository README for additional context
