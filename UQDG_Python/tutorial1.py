"""
Tutorial: Running Uncertainty Quantification (UQ) with xfoil and Monte Carlo Standard Error

This script demonstrates how to:
- Generate Monte Carlo samples for aerodynamic analysis using xfoil.
- Evaluate the solver with generated samples.
- Quantify uncertainty in the outputs using Monte Carlo standard error (SE).

Steps:
1. Define input parameter ranges for uniform sampling.
2. Generate sample input CSV file.
3. Run the solver (xfoil) on the samples.
4. Compute SE for mean, standard deviation, and P-box probability using UQ analysis.

NOTE: Python version automatically uses xfoil solver.
"""

import UQDGmxfoil.solver_eval as xmeval
import UQDGmxfoil.sample as smp
import UQDGmxfoil.uq_analysis as uq

# WARNING:
# Forced transition (xtr_upper, xtr_lower) as used in the UQDG challenge problem is available for xfoil.
# Python version automatically uses xfoil solver.

# Define the input parameters for the uniform distribution
input_names = ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower']

# Specify lower and upper bounds for each input parameter
xmin = [-0.3, 492500, -0.24, 0.255, 0.637]
xmax = [0.3, 507500, 0.24, 0.345, 0.763]

# Step 1: Create a sample input file with uniform distribution
# NOTE: Python version automatically uses xfoil solver
smp.sample(num_samples=100, 
           solver='xfoil').create_samples('uniform_samples.csv', 
                                          'uniform', 
                                          'monte', 
                                          xmin, 
                                          xmax, 
                                          input_names=input_names)

# Step 2: Evaluate the solver with the generated samples
xmeval.solver_eval('uniform_samples.csv').run()

# Step 3: Quantify the sample input uncertainty using Monte Carlo SE
SE = uq.uq_analysis().monte_carlo_SE('uniform_samples.csv')

# Step 4: Print the SE results for mean, standard deviation, and P-box probability
print("SE Mean:", SE[0])                  # Uncertainty interval for mean of Cl and Cm
print("SE Standard Deviation:", SE[1])    # Uncertainty interval for stddev of Cl and Cm
print("SE P-box Probability:", SE[2])     # Uncertainty interval for probability outside P-box

print("\nTutorial 1 completed successfully!")


