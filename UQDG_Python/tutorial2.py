import UQDGmxfoil.sample as smp
import UQDGmxfoil.solver_eval as xmeval
import UQDGmxfoil.poly_model as pm

# Define the input parameters for the uniform distribution
input_names = ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower']

xmin = [-0.3,492500,-0.24,0.255,0.637]
xmax = [0.3,507500,0.24,0.345,0.763]

# Create a training input file
smp.sample(num_samples=100, 
           solver='xfoil').create_samples('training_samples.csv', 
                                          'uniform', 
                                          'sobol', 
                                          xmin, 
                                          xmax, 
                                          input_names=input_names)

# Evaluate the solver with the generated samples
xmeval.solver_eval('training_samples.csv').run()

#Create a validation input file
smp.sample(num_samples=10000, 
           solver='xfoil').create_samples('validation_samples.csv', 
                                          'uniform', 
                                          'monte', 
                                          xmin, 
                                          xmax, 
                                          input_names=input_names)

#Evaluate the solver with the generated samples
xmeval.solver_eval('validation_samples.csv').run()

# Assemble the polynomial chaos surrogate model
poly = pm.poly_model()

poly.assemble_surrogate(
    in_sample='training_samples.csv',          # CSV file with input samples
    polynomial_degree=2,                        # Degree of polynomial chaos
    xmin=xmin,                                 # Minimum values for normalization
    xmax=xmax,                                 # Maximum values for normalization
    poly_coefficients_csv='poly_coeffs.csv'
)

poly.evaluate_surrogate(
    input_csv='validation_samples.csv',          # CSV file with input samples
    poly_c_csv='poly_coeffs.csv'                 # CSV file with saved coefficients
)