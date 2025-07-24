import numpy as np
import pandas as pd
import os
import math
import ast
from UQDGmxfoil.uq_analysis import *


class poly_model:
    """
    Polynomial Chaos Surrogate Model

    This class provides methods to assemble and evaluate polynomial chaos surrogates for aerodynamic analysis.
    It supports normalization of input data, robust coefficient extraction, and integration with UQ analysis.

    Key Features:
    - One-line surrogate assembly and evaluation
    - Normalization bounds (xmin, xmax) saved and loaded with coefficients
    - Uses pandas DataFrames for reliable input/output
    - Removes failed samples using UQ analysis
    - Vectorized evaluation of polynomial basis for all samples
    """
    def __init__(self):
        self.dir = os.getcwd()

    def basis(self, x, d):
        # Legendre polynomial basis up to degree 2
        b = [1, x, 0.5 * (3 * x ** 2 - 1)]
        return b[d]

    def poly_basis(self, in_sample):
        """
        Generates the polynomial basis vector for a given standardized input sample.
        Args:
            in_sample (np.ndarray): Standardized input sample (1D numpy array).
        Returns:
            np.ndarray: Vector of polynomial basis coefficients.
        """
        # Ensure input is numpy array
        in_sample = np.array(in_sample)
        pb = np.zeros(self.npb)
        pb[0] = 1  # Constant term

        # Non-cross terms: each variable, each polynomial degree
        idx = 1
        for degree in range(1, self.p + 1):
            for var in range(self.n):
                pb[idx] = self.basis(in_sample[var], degree)
                idx += 1

        # Cross-terms (only for p > 1): products of basis functions for variable pairs
        if self.p > 1:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    pb[idx] = self.basis(in_sample[i], 1) * self.basis(in_sample[j], 1)
                    idx += 1

        return pb

    def assemble_surrogate(self, csv_input, polynomial_degree, xmin, xmax, poly_coefficients_csv):
        """
        Assembles the polynomial chaos surrogate and saves coefficients and normalization bounds in one step.
        Args:
            csv_input (str): Input sample CSV filename.
            polynomial_degree (int): Degree of polynomial chaos interpolation.
            xmin (np.ndarray): Minimum values for each variable for normalization.
            xmax (np.ndarray): Maximum values for each variable for normalization.
            poly_coefficients_csv (str): Output CSV filename for coefficients.
        """
        # Ensure xmin/xmax are numpy arrays
        xmin = np.array(xmin)
        xmax = np.array(xmax)
        # Read input sample and determine n
        sample_df = pd.read_csv(self.dir + "/input/" + csv_input, delimiter=',', skiprows=1)
        self.n = sample_df.shape[1]
        self.p = polynomial_degree
        self.npb = int(math.factorial(self.n + self.p) / (math.factorial(self.n) * math.factorial(self.p)))
        # Standardize and assemble
        self.std_out(csv_input, xmin, xmax)
        self.assem(poly_coefficients_csv, xmin, xmax)

    def assem(self, poly_coefficients_csv, xmin=None, xmax=None):
        """
        Assembles polynomial coefficients from standardized input and output data, and saves them to CSV.
        Also saves normalization bounds as columns in the output CSV.
        Args:
            poly_coefficients_csv (str): Output CSV filename for coefficients.
            xmin (np.ndarray, optional): Minimum values for normalization.
            xmax (np.ndarray, optional): Maximum values for normalization.
        """
        # Read output sample
        self.out_sample = pd.read_csv(self.dir + '/output/' + self.name_sample[:-4] + '_out.csv', delimiter=',')
        Nsol = self.out_sample.shape[1]
        output_header = self.out_sample.columns.tolist()
        sol = np.zeros((self.npb, Nsol))
        # Loop over each output variable
        for j in range(Nsol):
            A = np.zeros((self.Ns, self.npb))
            b = self.out_sample[output_header[j]]
            # Build design matrix for least squares
            for i in range(self.Ns):
                A[i, :] = self.poly_basis(self.std_sample[i, :])
            # Solve least squares for coefficients
            sol[0:self.npb, j] = np.linalg.lstsq(A, b, rcond=None)[0]
        # Prepare header for output CSV
        if Nsol == 2:
            header = ['Cl_model', 'Cm_model']
        else:
            header = [f'out_model_{i+1}' for i in range(Nsol)]
        df = pd.DataFrame(sol, columns=header)
        out_path = self.dir + '/input/' + poly_coefficients_csv
        # Write normalization bounds and coefficients to CSV
        with open(out_path, 'w') as f:
            f.write(','.join([str(float(x)) for x in xmin]) + '\n')
            f.write(','.join([str(float(x)) for x in xmax]) + '\n')
        df.to_csv(out_path, mode='a', index=False)

    def std_out(self, in_sample_csv, xmin, xmax):
        """
        Loads and standardizes input data using provided normalization bounds.
        Removes failed samples using UQ analysis.
        Args:
            in_sample_csv (str): Input sample CSV filename.
            xmin (np.ndarray): Minimum values for normalization.
            xmax (np.ndarray): Maximum values for normalization.
        """
        # Ensure xmin/xmax are numpy arrays
        xmin = np.array(xmin)
        xmax = np.array(xmax)
        self.name_sample = in_sample_csv
        # Remove failed cases using UQ analysis
        uqa = uq_analysis()
        self.in_sample = uqa.removed_failed_cases(self.name_sample)
        self.Ns = self.in_sample.shape[0]
        self.std_sample = np.zeros(self.in_sample.shape)
        # Standardize each column
        for i, col in enumerate(self.in_sample.columns):
            self.std_sample[:, i] = (self.in_sample[col] - xmin[i]) / (xmax[i] - xmin[i])

    def evaluate_surrogate(self, csv_input, poly_c_csv):
        """
        Evaluates the polynomial chaos surrogate using coefficients and normalization bounds from CSV.
        Standardizes the input data and computes surrogate predictions for all samples.
        Args:
            input_csv (str): Input sample CSV filename.
            poly_c_csv (str): CSV file with surrogate coefficients and normalization bounds.
        """
        # Read normalization bounds from first two rows of coefficients CSV
        with open(self.dir + "/input/" + poly_c_csv, 'r') as f:
            lines = f.readlines()
            xmin = np.array([float(x) for x in lines[0].strip().split(',')])
            xmax = np.array([float(x) for x in lines[1].strip().split(',')])
        # Read coefficients
        pc = pd.read_csv(self.dir + "/input/" + poly_c_csv, skiprows=2, delimiter=',')
        header = list(pc.columns)
        coeff_cols = header
        pc_matrix = pc[coeff_cols].values
        # Standardize input samples
        self.std_out(csv_input, xmin, xmax)
        # Build polynomial basis for all samples
        PB = np.array([self.poly_basis(sample) for sample in self.std_sample])
        # Surrogate prediction
        sol = PB @ pc_matrix
        output_csv = self.dir + '/output/' + self.name_sample[:-4] + '_' + poly_c_csv[:-4] + "_sout.csv"
        # Write predictions to output CSV
        pd.DataFrame(sol, columns=header).to_csv(output_csv, index=False, header=True)