import numpy as np
import pandas as pd
import os
import ast
from UQDGmxfoil.uq_analysis import *
from scipy.optimize import curve_fit

def gaussian_variogram(magh, Co, C, a):
    """
    Gaussian variogram model.

    Args:
        magh (float or np.ndarray): Distance(s) between points.
        Co (float): Nugget parameter.
        C (float): Sill parameter.
        a (float): Range parameter.

    Returns:
        float or np.ndarray: Variogram value(s).
    """
    var = Co + C * (1 - np.exp(-(magh / a)**2))
    return var

class krig_model():
    """
    Kriging surrogate model for spatial interpolation and uncertainty quantification.

    Attributes:
        dir (str): Working directory.
        Ns (int): Number of samples.
        d (int): Input dimension.
        std_sample (np.ndarray): Standardized input samples.
        std_output (np.ndarray): Standardized output samples.
        out_sample (pd.DataFrame): Raw output samples.
        in_sample (pd.DataFrame): Raw input samples.
        varmodel (np.ndarray): Fitted variogram parameters.
        name_sample (str): Name of input sample file.
    """
    def __init__(self):
        """
        Initialize the krig_model object and its attributes.
        """
        self.dir = os.getcwd()
        self.Ns = None
        self.d = None
        self.std_sample = None
        self.std_output = None
        self.out_sample = None
        self.in_sample = None
        self.varmodel = None
        self.name_sample = None

    def variogram(self, out_var, num_bins):
        """
        Estimate variogram parameters for a given output variable.

        Args:
            out_var (int): Output variable index.
            num_bins (int): Number of bins for distance grouping.

        Sets:
            self.varmodel: Fitted variogram parameters [Co, C, a].
        """
        # Compute pairwise distances and semi-variances
        n = int((self.Ns*(self.Ns-1))/2 + 1)
        self.dist = np.zeros((n))
        svar = np.zeros(n)
        ii = 0
        for i in range(self.Ns):
            for j in range(i+1, self.Ns):
                ii += 1
                self.dist[ii] = np.linalg.norm(self.std_sample[i,:] - self.std_sample[j,:])
                svar[ii] = 0.5*(self.std_output[i,out_var] - self.std_output[j,out_var])**2

        # Bin distances and average semi-variances
        bins = np.linspace(0, np.max(self.dist), num_bins+1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        svar_bin = np.zeros(num_bins)
        for i in range(num_bins):
            mask = (self.dist >= bins[i]) & (self.dist < bins[i+1])
            if np.any(mask):
                svar_bin[i] = np.mean(svar[mask])
            else:
                svar_bin[i] = np.nan

        valid = ~np.isnan(svar_bin)
        h = bin_centers[valid]
        svar = svar_bin[valid]

        # Fit Gaussian variogram model to binned semi-variances
        popt, pconv = curve_fit(
            gaussian_variogram,
            h,
            svar,
            p0=[0.1, 1.0, 1.0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
        self.varmodel = popt

    def assemble_surrogate(self, training_data_file, sur_file, xmin, xmax, num_bins=5, gamma=1e-6, max_gamma=1e-2, cond_thresh=1e8):
        """
        Assemble and save the kriging surrogate model.

        Args:
            training_data_file (str): Input sample CSV filename.
            sur_file (str): Output surrogate filename.
            xmin (np.ndarray): Minimum bounds for normalization.
            xmax (np.ndarray): Maximum bounds for normalization.
            num_bins (int, optional): Number of bins for variogram calculation. Default is 5.
            gamma (float, optional): Initial regularization parameter. Default is 1e-6.
            max_gamma (float, optional): Maximum regularization parameter. Default is 1e-2.
            cond_thresh (float, optional): Condition number threshold for regularization. Default is 1e8.

        Output:
            Writes surrogate model data to sur_file in the input directory.
        """
        # Ensure xmin/xmax are numpy arrays
        xmin = np.array(xmin)
        xmax = np.array(xmax)
        # Standardize input/output data
        self.std_out(training_data_file, xmin, xmax)
        for ii in range(2):
            # Fit variogram for each output variable
            self.variogram(ii, num_bins)
            gamma = 1e-6

            # Assemble variogram matrix
            A = np.zeros((self.Ns + 1, self.Ns + 1), dtype=np.float32)
            for i in range(self.Ns):
                for j in range(self.Ns):
                    h = np.linalg.norm(self.std_sample[i, :] - self.std_sample[j, :])
                    A[i, j] = gaussian_variogram(h, self.varmodel[0], self.varmodel[1], self.varmodel[2])
            A[self.Ns, 0:self.Ns] = np.ones(self.Ns)
            A[0:self.Ns, self.Ns] = np.ones(self.Ns)
            A[-1, -1] = 0

            # Adaptive regularization to ensure invertibility
            while True:
                A_reg = A + np.eye(A.shape[0]) * gamma
                A_reg[-1, -1] = 0
                cond = np.linalg.cond(A_reg)
                if cond < cond_thresh or gamma >= max_gamma:
                    break
                gamma *= 10
            Ainv = np.linalg.inv(A_reg)

            # Save standardized samples and outputs
            if ii == 0:
                df_samples = pd.DataFrame(
                    np.hstack((self.std_sample, self.std_output)),
                    columns=[f'std_sample_{i+1}' for i in range(self.std_sample.shape[1])] + [f'std_output_{i+1}' for i in range(self.std_output.shape[1])]
                )
                with open(self.dir + '/input/' + sur_file, 'w') as f:
                    f.write(f"{self.Ns}\n")
                df_samples.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)
                # Save xmin and xmax as two rows, no header
                pd.DataFrame([list(map(float, xmin))]).to_csv(self.dir + '/input/' + sur_file, mode='a', index=False, header=False)
                pd.DataFrame([list(map(float, xmax))]).to_csv(self.dir + '/input/' + sur_file, mode='a', index=False, header=False)
            # Save model parameters and Ainv
            params = [self.out_sample.iloc[:, ii].mean(), self.out_sample.iloc[:, ii].std(), self.varmodel[0], self.varmodel[1], self.varmodel[2]]
            df_params = pd.DataFrame([params], columns=['mean', 'std', 'Co', 'C', 'a'])
            df_Ainv = pd.DataFrame(Ainv)
            df_params.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)
            df_Ainv.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False, header=False)

    def evaluate_surrogate(self, csv_input, sur_file):
        """
        Evaluate the kriging surrogate model on new input data.

        Args:
            csv_input (str): Input sample CSV filename.
            sur_file (str): Surrogate model filename.

        Output:
            Writes predictions to output CSV with columns ['cl', 'cm'].
        """
        # Read number of surrogate samples
        with open(self.dir + '/input/' + sur_file, 'r') as f:
            sur_Ns = int(f.readline().strip())

        # Read standardized samples and outputs
        df_samples = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=1, nrows=sur_Ns, header=0)
        std_input = df_samples.iloc[:, :self.d].values
        std_output = df_samples.iloc[:, self.d:self.d+2].values

        # Read xmin and xmax as two rows, no header
        norm_start = 1 + sur_Ns
        # Go down one more row to skip the extra row
        df_norm = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=norm_start + 1, nrows=2, header=None)
        xmin = np.array(df_norm.iloc[0, :self.d].astype(float))
        xmax = np.array(df_norm.iloc[1, :self.d].astype(float))

        # Standardize new input samples
        self.std_out(csv_input, xmin, xmax)

        sout = np.zeros((self.Ns, 2))  # Output array for predictions
        # Loop over each output variable
        current_row = 5 + sur_Ns
        for ii in range(2):
            # Read model parameters
            df_params = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=current_row, nrows=1, header=None)
            out_mean = df_params.iloc[0, 0]
            out_stddev = df_params.iloc[0, 1]
            varmodel = df_params.iloc[0, 2:].values
            current_row += 1
            # Read Ainv matrix (skip any header or non-numeric lines)
            with open(self.dir + '/input/' + sur_file, 'r') as f:
                lines = f.readlines()
            ainv_lines = lines[current_row:current_row + sur_Ns + 1]
            # Filter out lines that contain non-numeric values (e.g. headers)
            ainv_data = []
            for line in ainv_lines:
                values = line.strip().split(',')
                row = np.array([float(x) for x in values])
                ainv_data.append(row)
            Ainv = np.vstack(ainv_data)
            current_row += sur_Ns + 2

            # Compute distances between test points and surrogate input points
            diff = self.std_sample[:, None, :] - std_input[None, :, :]
            h = np.linalg.norm(diff, axis=2)
            b = gaussian_variogram(h, varmodel[0], varmodel[1], varmodel[2])

            # Build kriging weights
            b_full = np.ones((self.Ns, sur_Ns + 1))
            b_full[:, :-1] = b
            weights = Ainv @ b_full.T
            weights = weights.T

            # Compute surrogate predictions
            z = np.dot(weights[:, :sur_Ns], std_output[:, ii])
            sout[:, ii] = z * out_stddev + out_mean

        # Save predictions to output CSV
        df_sout = pd.DataFrame(sout, columns=['cl', 'cm'])
        df_sout.to_csv(self.dir + '/output/' + self.name_sample[:-4] + '_' + sur_file[:-4] +'_sout.csv', index=False, header=True)

    def std_out(self, in_sample, xmin, xmax):
        """
        Load and standardize input/output data.

        Args:
            in_sample (str): Input sample CSV filename.
            xmin (np.ndarray): Minimum bounds for normalization.
            xmax (np.ndarray): Maximum bounds for normalization.

        Sets:
            self.std_sample: Standardized input samples.
            self.std_output: Standardized output samples.
            self.in_sample: Cleaned input samples.
            self.out_sample: Raw output samples.
            self.Ns: Number of samples.
            self.d: Input dimension.
        """
        # Ensure xmin/xmax are numpy arrays
        xmin = np.array(xmin)
        xmax = np.array(xmax)
        self.name_sample = in_sample
        out_path = self.dir + "/output/" + self.name_sample[:-4] + "_out.csv"
        self.out_sample = pd.read_csv(out_path, delimiter=',')
        # Remove failed samples
        uqa = uq_analysis()
        self.in_sample = uqa.removed_failed_cases(self.name_sample)
        self.Ns = self.in_sample.shape[0]
        self.d = self.in_sample.shape[1]
        # Standardize input
        self.std_sample = (self.in_sample.values - xmin) / (xmax - xmin)
        # Standardize output
        self.std_output = np.zeros(self.out_sample.shape)
        for i in range(self.out_sample.shape[1]):
            col = self.out_sample.iloc[:, i]
            std = col.std()
            self.std_output[:, i] = (col - col.mean()) / std