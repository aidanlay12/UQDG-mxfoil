import UQDGmxfoil.solver_eval as xmeval
import UQDGmxfoil.sample as smp

# Define the input parameters for the uniform distribution
input_names = ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower']

xmin = [-0.3,492500,-0.24,0.255,0.637]
xmax = [0.3,507500,0.24,0.345,0.763]

# Create a sample input file
smp.sample(num_samples=100, 
           solver='mfoil').create_samples('uniform_samples.csv', 
                                          'uniform', 
                                          'sobol', 
                                          xmin, 
                                          xmax, 
                                          input_names=input_names)


# Evaluate the solver with the generated samples
xmeval.solver_eval('uniform_samples.csv').run()




