import numpy as np
import pandas as pd
import os
import math
import ast
from UQDG-mxfoil.uq_analysis import *


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

    def basis(self,x,d):
        #Uses the Legrende polynomials
        b = [1,x,0.5*(3*x**2 - 1)]
        return b[d]  
        
    def poly_basis(self, in_sample):
        """
        Generates the polynomial basis vector for a given standardized input sample.
        Args:
            in_sample (array-like): Standardized input sample.
        Returns:
            np.ndarray: Vector of polynomial basis coefficients.
        Raises:
            ValueError: If in_sample is not the expected length.
        """
        if self.n is None:
            raise ValueError("self.n must be set before calling poly_basis.")
        if len(in_sample) != self.n:
            raise ValueError(f"Input sample length {len(in_sample)} does not match expected n={self.n}.")

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
                for j in range(i, self.n):
                    pb[idx] = self.basis(in_sample[i], 1) * self.basis(in_sample[j], 1)
                    idx += 1

        return pb
                        

    def assemble_surrogate(self, in_sample, polynomial_degree, xmin, xmax, poly_coefficients_csv):
        """
        Assembles the polynomial chaos surrogate and saves coefficients and normalization bounds in one step.
        Args:
            in_sample (str): Input sample CSV filename.
            xmin (array-like): Minimum values for each variable for normalization.
            xmax (array-like): Maximum values for each variable for normalization.
            poly_coefficients_csv (str): Output CSV filename for coefficients.
        """
        self.p = polynomial_degree  # Degree of polynomial chaos interpolation
        # Determine n from input sample
        try:
            sample_df = pd.read_csv(self.dir + "/input/" + in_sample, delimiter=',')
        except Exception as e:
            raise IOError(f"Failed to read input CSV for n determination: {e}")
        self.n = sample_df.shape[1]
        self.npb = int(math.factorial(self.n + self.p) / (math.factorial(self.n) * math.factorial(self.p)))
        self.std_out(in_sample, xmin, xmax)
        self.assem(poly_coefficients_csv, xmin, xmax)

    def assem(self, poly_coefficients_csv, xmin=None, xmax=None):
        """
        Assembles polynomial coefficients from standardized input and output data, and saves them to CSV.
        Also saves normalization bounds as columns in the output CSV.
        Args:
            poly_coefficients_csv (str): Output CSV filename for coefficients.
            xmin (array-like, optional): Minimum values for normalization.
            xmax (array-like, optional): Maximum values for normalization.
        """
        try:
            self.out_sample = pd.read_csv(self.dir + '/output/' + self.name_sample[:-4] + '_out.csv', delimiter=',')
        except Exception as e:
            raise IOError(f"Failed to read output CSV: {e}")
        Nsol = self.out_sample.shape[1]
        output_header = self.out_sample.columns.tolist()
        sol = np.zeros((self.npb, Nsol))
        for j in range(Nsol):
            A = np.zeros((self.Ns, self.npb))
            b = self.out_sample[output_header[j]]
            for i in range(self.Ns):
                try:
                    A[i, :] = self.poly_basis(self.std_sample[i, :])
                except Exception as e:
                    raise ValueError(f"Error in poly_basis for sample {i}: {e}")
            try:
                sol[0:self.npb, j] = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception as e:
                raise ValueError(f"Least squares failed for output {output_header[j]}: {e}")
        if Nsol == 2:
            header = ['Cl_model', 'Cm_model']
        else:
            header = [f'out_model_{i+1}' for i in range(Nsol)]
        df = pd.DataFrame(sol, columns=header)
        if xmin is not None and xmax is not None:
            if not (isinstance(xmin, (np.ndarray, list)) and isinstance(xmax, (np.ndarray, list))):
                raise TypeError("xmin and xmax must be array-like.")
            if len(xmin) != self.n or len(xmax) != self.n:
                raise ValueError(f"xmin and xmax must have length n={self.n}.")
            df['xmin'] = xmin
            df['xmax'] = xmax
        try:
            df.to_csv(self.dir + '/input/' + poly_coefficients_csv, index=False)
        except Exception as e:
            raise IOError(f"Failed to write coefficients CSV: {e}")
    
    def std_out(self, in_sample, xmin, xmax):
        """
        Loads and standardizes input data using provided normalization bounds.
        Removes failed samples using UQ analysis.
        Args:
            in_sample (str): Input sample CSV filename.
            xmin (array-like): Minimum values for normalization.
            xmax (array-like): Maximum values for normalization.
        """
        try:
            self.name_sample = in_sample
            self.in_sample = pd.read_csv(self.dir + "/input/" + in_sample, delimiter=',')
        except Exception as e:
            raise IOError(f"Failed to read input CSV: {e}")
        try:
            uqa = uq_analysis()
            self.in_sample = uqa.removed_failed_cases(self.name_sample)
        except Exception as e:
            raise RuntimeError(f"Failed to remove failed cases: {e}")
        self.Ns = self.in_sample.shape[0]
        if not (isinstance(xmin, (np.ndarray, list)) and isinstance(xmax, (np.ndarray, list))):
            raise TypeError("xmin and xmax must be array-like.")
        if len(xmin) != self.n or len(xmax) != self.n:
            raise ValueError(f"xmin and xmax must have length n={self.n}.")
        self.std_sample = np.zeros(self.in_sample.shape)
        for i, col in enumerate(self.in_sample.columns):
            try:
                self.std_sample[:, i] = (self.in_sample[col] - xmin[i]) / (xmax[i] - xmin[i])
            except Exception as e:
                raise ValueError(f"Error normalizing column {col}: {e}")

    def evaluate_surrogate(self, input_csv, poly_c_csv):
        """
        Evaluates the polynomial chaos surrogate using coefficients and normalization bounds from CSV.
        Standardizes the input data and computes surrogate predictions for all samples.
        Args:
            input_csv (str): Input sample CSV filename.
            poly_c_csv (str): CSV file with surrogate coefficients and normalization bounds.
        """
        try:
            pc = pd.read_csv(self.dir + "/input/" + poly_c_csv)
        except Exception as e:
            raise IOError(f"Failed to read coefficients CSV: {e}")
        header = list(pc.columns)
        coeff_cols = [col for col in header if col not in ['xmin', 'xmax']]
        pc_matrix = pc[coeff_cols].values
        xmin = pc['xmin'].iloc[0] if 'xmin' in pc.columns else None
        xmax = pc['xmax'].iloc[0] if 'xmax' in pc.columns else None
        if xmin is not None and xmax is not None:
            xmin = np.array(ast.literal_eval(str(xmin))) if isinstance(xmin, str) else np.array(xmin)
            xmax = np.array(ast.literal_eval(str(xmax))) if isinstance(xmax, str) else np.array(xmax)
        self.std_out(input_csv, xmin, xmax)
        try:
            PB = np.array([self.poly_basis(sample) for sample in self.std_sample])
        except Exception as e:
            raise ValueError(f"Error computing polynomial basis: {e}")
        try:
            sol = PB @ pc_matrix
        except Exception as e:
            raise ValueError(f"Error in surrogate evaluation (matrix multiplication): {e}")
        output_csv = self.dir + '/output/' + self.name_sample[:-4] + '_' + poly_c_csv[:-4] + "_sout.csv"
        try:
            pd.DataFrame(sol).to_csv(output_csv, index=False, header=False)
        except Exception as e:
            raise IOError(f"Failed to write surrogate output CSV: {e}")