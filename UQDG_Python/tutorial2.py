"""
Tutorial: Polynomial Chaos Expansion Surrogate Modeling for Aerodynamic Analysis

This script demonstrates how to:
- Generate training and validation samples for xfoil.
- Evaluate the solver on both sets.
- Assemble a polynomial chaos surrogate model using training data.
- Evaluate the surrogate on validation data.
- Compute RMSE between surrogate predictions and actual solver outputs.

Steps:
1. Define input parameter ranges for uniform sampling.
2. Generate training samples (e.g., using Sobol sequence).
3. Run the solver on training samples.
4. Generate validation samples (e.g., using Monte Carlo).
   NOTE: If 'validation_samples.csv' already exists, comment out the creation line below to avoid appending to previous files.
5. Run the solver on validation samples.
6. Assemble and save polynomial chaos surrogate coefficients.
7. Evaluate surrogate predictions on validation samples.
8. Compute and print RMSE for surrogate accuracy.

NOTE: Python version automatically uses xfoil solver.
"""

import UQDGmxfoil.sample as smp
import UQDGmxfoil.solver_eval as xmeval
import UQDGmxfoil.poly_model as pm
import UQDGmxfoil.uq_analysis as uq

# Define the input parameters for the uniform distribution
input_names = ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower']

# Specify lower and upper bounds for each input parameter
xmin = [-0.3, 492500, -0.24, 0.255, 0.637]
xmax = [0.3, 507500, 0.24, 0.345, 0.763]

# Step 1: Create a training input file using Sobol sequence
# NOTE: Python version automatically uses xfoil solver
smp.sample(num_samples=50, 
           solver='xfoil').create_samples('training_samples_poly.csv', 
                                          'uniform', 
                                          'sobol', 
                                          xmin, 
                                          xmax, 
                                          input_names=input_names)

# Step 2: Evaluate the solver with the training samples
xmeval.solver_eval('training_samples_poly.csv').run()

# Step 3: Create a validation input file using Monte Carlo sampling
# WARNING: If 'validation_samples.csv' already exists, comment out the line below to avoid appending to previous files.
smp.sample(num_samples=100, 
           solver='xfoil').create_samples('validation_samples.csv', 
                                          'uniform', 
                                          'monte', 
                                          xmin, 
                                          xmax, 
                                          input_names=input_names)

# Step 4: Evaluate the solver with the validation samples
xmeval.solver_eval('validation_samples.csv').run()

# Step 5: Assemble the polynomial chaos surrogate model using training data
poly = pm.poly_model()
poly.assemble_surrogate(
    csv_input='training_samples_poly.csv',          # CSV file with input samples
    polynomial_degree=2,                            # Degree of polynomial chaos
    xmin=xmin,                                      # Minimum values for normalization
    xmax=xmax,                                      # Maximum values for normalization
    poly_coefficients_csv='poly_coeffs.csv'         # Output CSV for surrogate coefficients
)

# Step 6: Evaluate the surrogate model on validation samples
poly.evaluate_surrogate(
    csv_input='validation_samples.csv',             # CSV file with validation input samples
    poly_c_csv='poly_coeffs.csv'                    # CSV file with saved coefficients
)

# Step 7: Compute RMSE between surrogate predictions and actual outputs
RMSE = uq.uq_analysis().surrogate_RMSE(
    csv_file_out='validation_samples_out.csv',
    sur_file_out='validation_samples_poly_coeffs_sout.csv'
)

# Step 8: Print RMSE for Cl and Cm
print("Cl surrogate RMSE:", RMSE[0])
print("Cm surrogate RMSE:", RMSE[1])

print("\nTutorial 2 completed successfully!")