import os
from scipy.stats.qmc import Sobol, LatinHypercube
from scipy.stats import norm
import numpy as np
import pandas as pd

class sample:
    """
    Unified sampler for generating input samples for various distributions and sampling methods.

    Attributes:
        Ns (int): Number of samples.
        solver (str): Solver type.
        d (int): Number of input variables.
        fx (np.ndarray): Generated samples.
        dir (str): Working directory.
    """
    def __init__(self, num_samples=50, solver='xfoil'):
        """
        Initialize the sample generator.

        Args:
            num_samples (int, optional): Number of samples. Default is 50.
            solver (str, optional): Solver type. Default is 'xfoil'.
        """
        self.Ns = num_samples
        self.solver = solver
        self.d = None
        self.fx = None
        self.dir = os.getcwd()

    def create_samples(self, csv_name, dist_type, method, param1, param2, gci_panels=None, input_names=None, num_samples=None, solver=None):
        """
        Create samples and save them to a CSV file.

        Args:
            csv_name (str): Output CSV filename.
            dist_type (str): 'normal', 'uniform', 'epistemic', or 'gci'.
            method (str): 'sobol', 'lhs', 'monte', 'factorial' (ignored for 'gci').
            param1: For 'normal', mean array; for 'uniform' and 'epistemic', xmin array; for 'gci', input values array.
            param2: For 'normal', std array; for 'uniform' and 'epistemic', xmax array; for 'gci', ignored.
            gci_panels (list or np.ndarray, optional): For 'gci', list of panel numbers for each sample.
            num_samples (int, optional): Override number of samples for this call.
            solver (str, optional): Override solver type for this call.
            input_names (list): List of input variable names.
        """
        # Set sample size and solver if provided
        if num_samples is not None:
            self.Ns = num_samples
        if solver is not None:
            self.solver = solver
        # Handle GCI mode
        if dist_type == 'gci':
            self.d = len(param1)
            self.fx = np.tile(param1, (self.Ns, 1))
            self.gci_panels = np.array(gci_panels).reshape(-1, 1)
            self.save_samples(csv_name, gci_mode=True, input_names=input_names)
        else:
            # Generate samples for other distributions
            self.fx = self.sample(dist_type, method, param1, param2)
            self.save_samples(csv_name, input_names=input_names)

    def save_samples(self, csv_name, input_names=None, gci_mode=False):
        """
        Save generated samples to a CSV file with two headers.

        Args:
            csv_name (str): Output CSV filename.
            input_names (list, optional): List of input variable names.
                Defaults to ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower'].
            gci_mode (bool): If True, append 'panel_size' column for GCI study.
        """
        # Set default input names if not provided
        if input_names is None:
            input_names = ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower']
        header1 = [self.solver, self.d, self.Ns]
        header2 = input_names.copy()
        # Add panel_size column for GCI mode
        if gci_mode:
            header2.append('panel_size')
            df = pd.DataFrame(np.hstack([self.fx, self.gci_panels]), columns=header2)
        else:
            df = pd.DataFrame(self.fx, columns=header2)
        out_path = self.dir + '/input/' + csv_name
        # Write headers and data to CSV
        with open(out_path, 'w', newline='') as f:
            f.write(','.join(map(str, header1)) + '\n')
            f.write(','.join(header2) + '\n')
        df.to_csv(out_path, mode='a', index=False, header=False)

    def sample(self, dist_type, method, param1, param2):
        """
        Generate samples for a specified distribution and method.

        Args:
            dist_type (str): 'normal', 'uniform', or 'epistemic'
            method (str): 'sobol', 'lhs', 'monte', or 'factorial'
            param1: For 'normal', mean array; for 'uniform' and 'epistemic', xmin array.
            param2: For 'normal', std array; for 'uniform' and 'epistemic', xmax array.

        Returns:
            np.ndarray: Generated samples.
        """
        # Set distribution parameters and dimension
        if dist_type == 'normal':
            mean = np.array(param1)
            std = np.array(param2)
            self.d = mean.size
        elif dist_type == 'uniform':
            xmin = np.array(param1)
            xmax = np.array(param2)
            self.d = xmin.size
        elif dist_type == 'epistemic':
            xmin = np.array(param1)
            xmax = np.array(param2)
            self.d = xmin.size
            # Factorial grid for epistemic
            if method == 'factorial':
                n = int(self.Ns ** (1 / self.d))
                ls = np.zeros((n, self.d))
                for i in range(self.d):
                    ls[:, i] = np.linspace(xmin[i], xmax[i], n)
                fx = np.zeros((n ** self.d, self.d))
                zz = 0
                # Nested loops for grid sampling
                for i in range(n):
                    for j in range(n):
                        for z in range(n):
                            for ii in range(n):
                                for jj in range(n):
                                    fx[zz, 0:5] = [ls[i, 0], ls[j, 1], ls[z, 2], ls[ii, 3], ls[jj, 4]]
                                    zz += 1
                self.fx = fx
                return self.fx
        # Generate standard uniform samples
        if method == 'sobol':
            su = Sobol(self.d).random(self.Ns)
        elif method == 'lhs':
            su = LatinHypercube(self.d).random(self.Ns)
        elif method == 'monte':
            su = np.random.rand(self.Ns, self.d)
        # Transform samples according to distribution
        if dist_type == 'normal':
            fx = norm.ppf(su)
            for i in range(self.d):
                fx[:, i] = std[i] * fx[:, i] + mean[i]
            self.fx = fx
        elif dist_type == 'uniform':
            fx = np.zeros((self.Ns, self.d))
            for i in range(self.d):
                fx[:, i] = (xmax[i] - xmin[i]) * su[:, i] + xmin[i]
            self.fx = fx
        return self.fx

    def create_gci_samples(self, csv_name, evaluation_point, panel_sizes, input_names=None, num_samples=None, solver=None):
        """
        Create and save samples for a GCI study, where all input parameters are the same
        and only the panel size varies for each sample.

        Args:
            csv_name (str): Output CSV filename.
            evaluation_point (array-like): Input parameter values (length d).
            panel_sizes (array-like): Panel size for each sample (length Ns).
            input_names (list, optional): List of input variable names.
                Defaults to ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower'].
            num_samples (int, optional): Override number of samples for this call.
            solver (str, optional): Override solver type for this call.
        """
        # Set sample size and solver if provided
        if num_samples is not None:
            self.Ns = num_samples
        else:
            self.Ns = len(panel_sizes)
        if solver is not None:
            self.solver = solver
        self.d = len(evaluation_point)
        if input_names is None:
            input_names = ['alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower']
        # Create repeated input values and append panel sizes
        fx = np.tile(evaluation_point, (self.Ns, 1))
        panel_sizes = np.array(panel_sizes).reshape(-1, 1)
        header1 = [self.solver, self.d, self.Ns]
        header2 = input_names + ['panel_size']
        df = pd.DataFrame(np.hstack([fx, panel_sizes]), columns=header2)
        out_path = self.dir + '/input/' + csv_name
        # Write headers and data to CSV
        with open(out_path, 'w', newline='') as f:
            f.write(','.join(map(str, header1)) + '\n')
            f.write(','.join(header2) + '\n')
        df.to_csv(out_path, mode='a', index=False, header=False)
    
    def mix_krig(self, csv_name, xmin, xmax, epi_Ns=32, input_names=None, num_samples=None, solver=None):
        """
        Generate a mixed training sample for kriging model by combining
        a Sobol uniform sample and an epistemic factorial sample.

        Args:
            csv_name (str): Output CSV filename.
            xmin (array-like): Minimum values for uniform distribution.
            xmax (array-like): Maximum values for uniform distribution.
            epi_Ns (int, optional): Number of epistemic samples (factorial grid points per dimension).
            input_names (list, optional): Input variable names for saving.
            num_samples (int, optional): Override number of samples for this call.
            solver (str, optional): Override solver type for this call.

        Sets:
            self.fx: Combined sample array.
            self.Ns: Total number of samples.
            self.d: Number of input variables.
        """
        # Set sample size and solver if provided
        if num_samples is not None:
            self.Ns = num_samples
        if solver is not None:
            self.solver = solver
        # Generate Sobol and epistemic samples
        self.Ns = self.Ns-epi_Ns
        sobol_fx = self.sample('uniform', 'sobol', xmin, xmax)
        self.Ns = epi_Ns
        epi_fx = self.sample('epistemic', 'factorial', xmin, xmax)
        # Combine samples and save
        self.fx = np.vstack((sobol_fx, epi_fx))
        self.Ns = self.fx.shape[0]
        self.d = self.fx.shape[1]
        if input_names is not None:
            self.save_samples(csv_name, input_names=input_names)