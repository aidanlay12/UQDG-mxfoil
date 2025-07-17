import numpy as np
import pandas as pd
import os
from UQDGmxfoil.uq_analysis import *

def gaussian_variogram(magh,Co,C,a):
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
        varmodel (np.ndarray): Variogram model parameters.
        name_sample (str): Name of the sample file.
    """
    def __init__(self):
        """
        Initialize the krig_model class and its attributes.
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

        Raises:
            RuntimeError: If curve fitting fails.
        """
        #Computes the distance from the point being evaulated
        n = int((self.Ns*(self.Ns-1))/2 + 1)
        self.dist = np.zeros((n))
        svar = np.zeros(n)
        
        ii = 0
        for i in range(self.Ns):
            for j in range(i+1, self.Ns):
                ii = ii + 1
                self.dist[ii] = np.linalg.norm(self.std_sample[i,:] - self.std_sample[j,:])
                svar[ii] = 0.5*(self.std_output[i,out_var] - self.std_output[j,out_var])**2
        
        bins = np.linspace(0,np.max(self.dist),num_bins+1)
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
        

        #Curve fits the semi-variance values to the guassian model
        try:
            popt, pconv = curve_fit(
                gaussian_variogram,
                h,
                svar,
                p0=[0.1, 1.0, 1.0],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),  # All parameters >= 0
                maxfev=10000
            )
        except Exception as e:
            raise RuntimeError(f"Variogram curve fitting failed: {e}")
        self.varmodel = popt #Saves the model parameters


    def assem(self, training_data_file, sur_file, xmin=None, xmax=None, num_bins=5, gamma=1e-6, max_gamma=1e-2, cond_thresh=1e8):
        """
        Assemble and save the kriging surrogate model.

        Args:
            training_data_file (str): Input sample CSV filename.
            sur_file (str): Output surrogate filename.
            xmin (array-like): Minimum bounds for normalization.
            xmax (array-like): Maximum bounds for normalization.
            num_bins (int, optional): Number of bins for variogram calculation. Default is 5.
            gamma (float, optional): Initial regularization parameter. Default is 1e-6.
            max_gamma (float, optional): Maximum regularization parameter. Default is 1e-2.
            cond_thresh (float, optional): Condition number threshold for regularization. Default is 1e8.

        Raises:
            RuntimeError: If standardization, variogram, matrix inversion, or file writing fails.
        """
        # Loads the output data and writes all results using pandas
        try:
            self.std_out(training_data_file, xmin, xmax)  # Standardize the input data
        except Exception as e:
            raise RuntimeError(f"Failed to standardize input/output: {e}")
        for ii in range(2):
            try:
                self.variogram(ii, num_bins)  # Creates a variogram
            except Exception as e:
                raise RuntimeError(f"Variogram computation failed for output {ii}: {e}")
            gamma = 1e-6  # Regularization parameter to avoid singular matrix

            # Assemble variogram matrix
            A = np.zeros((self.Ns + 1, self.Ns + 1), dtype=np.float32)
            for i in range(self.Ns):
                for j in range(self.Ns):
                    h = np.linalg.norm(self.std_sample[i, :] - self.std_sample[j, :])
                    A[i, j] = gaussian_variogram(h, self.varmodel[0], self.varmodel[1], self.varmodel[2])
            A[self.Ns, 0:self.Ns] = np.ones(self.Ns)
            A[0:self.Ns, self.Ns] = np.ones(self.Ns)
            A[-1, -1] = 0

            # Adaptive regularization
            while True:
                A_reg = A + np.eye(A.shape[0]) * gamma
                A_reg[-1, -1] = 0
                cond = np.linalg.cond(A_reg)
                if cond < cond_thresh or gamma >= max_gamma:
                    break
                gamma *= 10
            print(gamma)
            try:
                Ainv = np.linalg.inv(A_reg)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Matrix inversion failed: {e}")

            # Prepare DataFrames for output
            # First block: sample data
            if ii == 0:
                df_samples = pd.DataFrame(
                    np.hstack((self.std_sample, self.std_output)),
                    columns=[f'std_sample_{i+1}' for i in range(self.std_sample.shape[1])] + [f'std_output_{i+1}' for i in range(self.std_output.shape[1])]
                )
                # Save number of samples as a separate file or as a header row (pandas does not support header rows)
                with open(self.dir + '/input/' + sur_file, 'w') as f:
                    f.write(f"{self.Ns}\n")
                try:
                    df_samples.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)
                except Exception as e:
                    raise RuntimeError(f"Failed to write sample data: {e}")

                # Save normalization bounds as a separate block
                if xmin is not None and xmax is not None:
                    df_norm = pd.DataFrame({'xmin': [list(xmin)], 'xmax': [list(xmax)]})
                    try:
                        df_norm.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)
                    except Exception as e:
                        raise RuntimeError(f"Failed to write normalization bounds: {e}")

            # Second block: model parameters and Ainv
            params = [self.out_sample[:, ii].mean(), self.out_sample[:, ii].std(), self.varmodel[0], self.varmodel[1], self.varmodel[2]]
            df_params = pd.DataFrame([params], columns=['mean', 'std', 'Co', 'C', 'a'])
            df_Ainv = pd.DataFrame(Ainv)
            try:
                df_params.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)
                df_Ainv.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False, header=False)
            except Exception as e:
                raise RuntimeError(f"Failed to write model parameters or Ainv: {e}")
    

    def eval(self, csv_input, sur_file):
        """
        Evaluate the kriging surrogate model on new input data.

        Args:
            csv_input (str): Input sample CSV filename.
            sur_file (str): Surrogate model filename.

        Raises:
            RuntimeError: If reading files, standardization, kriging computation, or saving predictions fails.
        Notes:
            If vectorized kriging computation fails due to memory or shape issues,
            falls back to a non-vectorized loop for each sample.
        """
        # Read the number of samples (first line)
        try:
            with open(self.dir + '/input/' + sur_file, 'r') as f:
                sur_Ns = int(f.readline().strip())
        except Exception as e:
            raise RuntimeError(f"Failed to read surrogate file: {e}")

        # Read the sample data block
        try:
            df_samples = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=1, nrows=sur_Ns)
        except Exception as e:
            raise RuntimeError(f"Failed to read sample data block: {e}")

        std_input = df_samples.iloc[:, :self.d].values
        std_output = df_samples.iloc[:, self.d:].values

        # Read normalization bounds (next row after samples)
        try:
            df_norm = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=1 + sur_Ns, nrows=1)
            xmin = ast.literal_eval(df_norm['xmin'].iloc[0])
            xmax = ast.literal_eval(df_norm['xmax'].iloc[0])
        except Exception as e:
            raise RuntimeError(f"Failed to read normalization bounds: {e}")

        try:
            self.std_out(csv_input, xmin, xmax)  # Standardize the input data
        except Exception as e:
            raise RuntimeError(f"Failed to standardize input for eval: {e}")

        sout = np.zeros((self.Ns, 2))  # Output array for predictions

        # Track the current row for reading parameters and Ainv
        current_row = 1 + sur_Ns
        for ii in range(2):
            # Read model parameters
            try:
                df_params = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=current_row, nrows=1)
                out_mean = df_params.iloc[0, 0]
                out_stddev = df_params.iloc[0, 1]
                varmodel = df_params.iloc[0, 2:].values
            except Exception as e:
                raise RuntimeError(f"Failed to read model parameters: {e}")

            current_row += 1
            # Read Ainv
            try:
                df_Ainv = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=current_row, nrows=sur_Ns + 1, header=None)
                Ainv = df_Ainv.values
            except Exception as e:
                raise RuntimeError(f"Failed to read Ainv: {e}")
            current_row += sur_Ns + 1

            # Vectorized distance computation between all test points and sur_input points
            try:
                diff = self.std_sample[:, None, :] - std_input[None, :, :]
                h = np.linalg.norm(diff, axis=2)
                b = gaussian_variogram(h, varmodel[0], varmodel[1], varmodel[2])

                b_full = np.ones((self.Ns, sur_Ns + 1))
                b_full[:, :-1] = b

                weights = Ainv @ b_full.T
                weights = weights.T

                z = np.dot(weights[:, :sur_Ns], std_output[:, ii])
                sout[:, ii] = z * out_stddev + out_mean
            except Exception as e:
                print(f"Vectorized kriging failed: {e}. Falling back to non-vectorized computation.")
                # Non-vectorized fallback
                for k in range(self.Ns):
                    try:
                        # Compute distances for each test point
                        h_row = np.array([np.linalg.norm(self.std_sample[k, :] - std_input[j, :]) for j in range(sur_Ns)])
                        b_row = gaussian_variogram(h_row, varmodel[0], varmodel[1], varmodel[2])
                        b_full_row = np.ones(sur_Ns + 1)
                        b_full_row[:-1] = b_row
                        weights_row = Ainv @ b_full_row
                        z = np.dot(weights_row[:sur_Ns], std_output[:, ii])
                        sout[k, ii] = z * out_stddev + out_mean
                    except Exception as e2:
                        raise RuntimeError(f"Non-vectorized kriging failed at sample {k}: {e2}")

        # Save predictions using pandas
        try:
            df_sout = pd.DataFrame(sout)
            df_sout.to_csv(self.dir + '/output/' + self.name_sample[:-4] + '_' + sur_file[:-4] +'_sout.csv', index=False, header=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save predictions: {e}")

    def std_out(self, in_sample, xmin, xmax):
        """
        Load and standardize input/output data.

        Args:
            in_sample (str): Input sample CSV filename.
            xmin (array-like): Minimum bounds for normalization.
            xmax (array-like): Maximum bounds for normalization.

        Raises:
            ValueError: If bounds are missing or input sample is empty.
            FileNotFoundError: If output file is missing.
        """
        # Loads and standardizes input/output data using pandas, expects headers in CSVs
        if xmin is None or xmax is None:
            raise ValueError("xmin and xmax must be provided for standardization.")
        self.name_sample = in_sample
        out_path = self.dir + "/output/" + self.name_sample[:-4] + "_out.csv"
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Output file not found: {out_path}")
        self.out_sample = pd.read_csv(out_path, delimiter=',')

        # Remove failed samples
        uqa = uq_analysis()
        self.in_sample = uqa.removed_failed_cases(self.name_sample)

        self.Ns = self.in_sample.shape[0]
        self.d = self.in_sample.shape[1]
        if self.Ns == 0 or self.d == 0:
            raise ValueError("Input sample is empty or has invalid shape.")

        # Standardize input
        self.std_sample = (self.in_sample.values - np.array(xmin)) / (np.array(xmax) - np.array(xmin))

        # Standardize output
        self.std_output = np.zeros(self.out_sample.shape)
        for i in range(self.out_sample.shape[1]):
            col = self.out_sample.iloc[:, i]
            std = col.std()
            if std == 0:
                raise ValueError(f"Standard deviation for output column {i} is zero.")
            self.std_output[:, i] = (col - col.mean()) / std