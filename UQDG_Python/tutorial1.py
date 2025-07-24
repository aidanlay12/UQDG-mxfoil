"""
Tutorial: Running Uncertainty Quantification (UQ) with XFOIL/mfoil and Monte Carlo Standard Error

This script demonstrates how to:
- Generate Monte Carlo samples for aerodynamic analysis using XFOIL or mfoil.
- Evaluate the solver with generated samples.
- Quantify uncertainty in the outputs using Monte Carlo standard error (SE).

Steps:
1. Define input parameter ranges for uniform sampling.
2. Generate sample input CSV file.
3. Run the solver (XFOIL or mfoil) on the samples.
4. Compute SE for mean, standard deviation, and P-box probability using UQ analysis.

Modify the 'solver' argument in smp.sample(...) to 'xfoil' or 'mfoil' as needed.
"""

import UQDGmxfoil.solver_eval as xmeval
import UQDGmxfoil.sample as smp
import UQDGmxfoil.uq_analysis as uq

# WARNING:
# Forced transition (xtr_upper, xtr_lower) as used in the UQDG challenge problem is NOT available for the Python version of mfoil.
# If you require forced transition, use the MATLAB version of mfoil.

# Define the input parameters for the uniform distribution
input_names = ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower']

# Specify lower and upper bounds for each input parameter
xmin = [-0.3, 492500, -0.24, 0.255, 0.637]
xmax = [0.3, 507500, 0.24, 0.345, 0.763]

# Step 1: Create a sample input file with uniform distribution
# Change solver='xfoil' to solver='mfoil' to use mfoil instead
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


