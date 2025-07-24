# UQDG-mxfoil

An educational uncertainty quantification (UQ) software library for aerodynamic analysis with specialized implementations for different solvers. This library provides tools for sampling, surrogate modeling, and uncertainty analysis with both Python and MATLAB implementations.

## Overview

UQDG-mxfoil enables uncertainty quantification workflows for computational fluid dynamics (CFD) analysis with solver-specific implementations:
- **Python Implementation**: XFOIL solver for 2D airfoil analysis
- **MATLAB Implementation**: mfoil solver for multi-element airfoil analysis

The library supports various UQ methodologies including Monte Carlo sampling, surrogate modeling (Polynomial Chaos and Kriging), and statistical analysis with uncertainty bounds.

## Repository Structure

```
UQDG-mxfoil/
├── UQDG_Python/          # Python implementation (XFOIL)
├── UQDG_MATLAB/          # MATLAB implementation (mfoil)
├── LICENSE               # MIT License
├── CHANGELOG.md          # Version history
├── CONTRIBUTING.md       # Contribution guidelines
└── README.md             # This file
```

## Python Implementation (UQDG_Python/) - XFOIL Solver

### Features
- **Sampling Methods**: Monte Carlo, Latin Hypercube, Sobol sequences
- **Surrogate Models**: Polynomial Chaos Expansion, Kriging
- **Solver**: XFOIL integration for 2D airfoil analysis
- **UQ Analysis**: Monte Carlo standard error, Grid Convergence Index (GCI)
- **I/O**: CSV-based input/output with double-header format

### Installation

The Python package can be installed using pip:

```bash
cd UQDG_Python/
pip install .
```

**Requirements**: Python ≥3.9, numpy, scipy, matplotlib, pandas

### Core Modules

- **`sample.py`**: Unified sampler for various probability distributions
- **`solver_eval.py`**: Aerodynamic solver interface (XFOIL)
- **`uq_analysis.py`**: Uncertainty quantification and statistical analysis
- **`poly_model.py`**: Polynomial Chaos surrogate modeling
- **`krig_model.py`**: Kriging surrogate modeling

### Tutorial Overview

The Python implementation includes four comprehensive tutorials that demonstrate different UQ workflows:

- **Tutorial 1**: Introduces basic uncertainty quantification using Monte Carlo sampling. Covers generating uniform samples, running XFOIL simulations, and computing Monte Carlo standard errors for statistical uncertainty bounds.

- **Tutorial 2**: Demonstrates Polynomial Chaos Expansion surrogate modeling. Shows how to generate training samples using Sobol sequences, create validation datasets, build polynomial surrogate models, and evaluate surrogate accuracy using RMSE metrics.

- **Tutorial 3**: Focuses on Kriging surrogate modeling techniques. Covers training Kriging models on aerodynamic data, making predictions on validation sets, and assessing surrogate model performance for efficient uncertainty propagation.

- **Tutorial 4**: Explores Grid Convergence Index (GCI) analysis for numerical uncertainty quantification. Demonstrates mesh refinement studies, convergence analysis, and estimation of discretization errors in CFD simulations.

## MATLAB Implementation (UQDG_MATLAB/) - mfoil Solver

### Features
- **mfoil Solver Integration**: Multi-element airfoil analysis capabilities
- **Complete mfoil Support**: NACA airfoils, flap deflection, operating conditions
- **Advanced Sampling**: Monte Carlo, Latin Hypercube, Sobol sequences
- **Surrogate Modeling**: Polynomial Chaos and Kriging implementations
- **Robust Error Handling**: Failed simulation tracking and reporting

### Core Functions

- **`sample.m`**: Multi-distribution sampler with various sequence types
- **`solver_eval.m`**: mfoil solver interface for multi-element airfoils
- **`uq_analysis.m`**: Statistical analysis and uncertainty quantification
- **`poly_model.m`**: Polynomial Chaos surrogate implementation
- **`krig_model.m`**: Kriging surrogate implementation

### Key Capabilities

#### solver_eval.m Features
- **mfoil integration**: Complete solver with multi-element airfoil support
- **NACA airfoil support**: Built-in NACA airfoil generation and modification
- **Flap deflection**: Advanced control surface modeling capabilities
- **Error handling**: Robust failed simulation management
- **Progress tracking**: Real-time progress with time estimates
- **CSV I/O**: Double-header format with automatic output

## Input/Output Format

### CSV Input Format (Double Header)
```
solver,num_variables,num_samples
alpha,Re,flap_deflection,xtr_upper,xtr_lower
-0.1,500000,0.1,0.3,0.7
0.0,500000,0.0,0.3,0.7
...
```

**Solver Options:**
- **`xfoil`**: Use XFOIL solver (Python implementation only)
- **`mfoil`**: Use mfoil solver (MATLAB implementation only)

### Output Files
- **`*_out.csv`**: Solver results (Cl, Cm columns)
- **`*_fail.csv`**: Failed cases with input parameters
- **`*_sout.csv`**: Surrogate model predictions

## Tutorials

**Python Implementation (XFOIL):**
1. **Tutorial 1**: Monte Carlo sampling and basic UQ with standard error analysis
2. **Tutorial 2**: Polynomial Chaos surrogate modeling with training and validation
3. **Tutorial 3**: Kriging surrogate modeling with RMSE evaluation
4. **Tutorial 4**: Grid Convergence Index (GCI) analysis for numerical uncertainty

**MATLAB Implementation (mfoil):**
1. **Tutorial 1**: Monte Carlo sampling and basic UQ with mfoil solver
2. **Tutorial 2**: Polynomial Chaos surrogate modeling for multi-element airfoils
3. **Tutorial 3**: Kriging surrogate modeling with multi-element configurations

## Applications

- Uncertainty quantification in aerodynamic design
- Surrogate-based optimization
- Sensitivity analysis
- Model validation and verification
- Educational CFD/UQ workflows

## Author

**Aidan Lay**  
Email: alay12@vols.utk.edu  
University of Tennessee, Knoxville

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
