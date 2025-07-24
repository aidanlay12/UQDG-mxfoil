import os
import numpy as np
import pandas as pd
from cfdverify.discretization import Classic
from cfdverify.utils import mesh_size


class uq_analysis:
    """
    Performs uncertainty quantification (UQ) analysis on solver output data.

    This class provides methods for:
    - Removing failed cases from input data.
    - Calculating probability of outputs outside a specified P-box.
    - Estimating uncertainty using Monte Carlo standard error.
    - Performing GCI (Grid Convergence Index) analysis.
    - Computing RMSE between surrogate predictions and actual outputs.

    Attributes:
        dir (str): Working directory.
        nout (int): Number of output variables.
        Np (int): Number of samples.
        CMlim (np.ndarray): Moment coefficient limits for P-box.
        CLlim (np.ndarray): Lift coefficient limits for P-box.
    """

    def __init__(self):
        """
        Initialize the uq_analysis object and its attributes.
        """
        self.dir = os.getcwd()
        self.nout = 0
        self.Np = 0
        self.CMlim = np.float64([-0.05, -0.044])  # Moment coefficient range for P-box in challenge problem
        self.CLlim = np.float64([0.15, 0.27])     # Lift coefficient range for P-box in challenge problem

    def removed_failed_cases(self, csv_file):
        """
        Reads the input data from the solver and removes failed cases.

        Args:
            csv_file (str): Input CSV filename.

        Returns:
            pd.DataFrame: Input data with failed cases removed.
        """
        # Loads the input data from the solver (skipping header row)
        df = pd.read_csv(self.dir + '/input/' + csv_file, delimiter=',', skiprows=1)
        fail_path = self.dir + '/output/' + csv_file[:-4] + '_fail.csv'
        # Remove failed cases from input DataFrame if fail file exists
        if os.path.exists(fail_path):
            df_fail = pd.read_csv(fail_path, delimiter=',')
            if 'Index' in df_fail.columns:
                failed_indices = df_fail['Index'].to_numpy()
                # Remove rows with these indices from df
                df = df.drop(index=failed_indices)
                df = df.reset_index(drop=True)
        return df

    def outside_pbox_prob(self, data):
        """
        Calculates the percentage of samples outside the P-box bounds for lift and moment coefficients.

        Args:
            data (np.ndarray): Array of output samples (columns: cl, cm).

        Returns:
            float: Percentage of samples outside the P-box.
        """
        cl_out = (data[:, 0] < self.CLlim[0]) | (data[:, 0] > self.CLlim[1])
        cm_out = (data[:, 1] < self.CMlim[0]) | (data[:, 1] > self.CMlim[1])
        nout = np.count_nonzero(cl_out | cm_out)
        Pi = (nout / data.shape[0]) * 100
        return Pi

    def monte_carlo_SE(self, csv_file, confidence_level_z=1.96):
        """
        Calculates uncertainty intervals for mean, standard deviation, and P-box probability using Monte Carlo method.

        Args:
            csv_file (str): Input CSV filename.
            confidence_level_z (float): Z-value for confidence interval (default is 1.96 for 95% CI).

        Returns:
            list: [mean uncertainty interval (array), std uncertainty interval (array), P-box uncertainty interval (float)]
        """
        # Loads the output data from the solver
        df = pd.read_csv(self.dir + '/output/' + csv_file[:-4] + '_out.csv', delimiter=',')
        Np = len(df)  # Number of samples
        z = confidence_level_z  # 95% confidence interval constant
        Ci_err = np.zeros(2)
        stdd_err = np.zeros(2)
        # Calculate uncertainty intervals for mean and stddev for Cl and Cm
        for idx, i in enumerate(['Cl', 'Cm']):
            stddev = np.std(df[i], ddof=1)
            conv = 1 / np.sqrt(Np)
            SE = stddev * conv
            Ci_err[idx] = z * SE
            mu = np.mean(df[i])
            mu4 = np.mean((df[i] - mu) ** 4)
            SE = conv * np.sqrt((mu4 - (stddev ** 4) * (Np - 3) / (Np - 1)))
            SE = SE / (2 * stddev)
            stdd_err[idx] = z * SE
        # Calculate P-box probability and uncertainty
        out = df.to_numpy()[:, :]
        Pmean = self.outside_pbox_prob(out)
        Pstddev = np.sqrt(Pmean * (100 - Pmean))
        SE = Pstddev * conv
        P_err = z * SE
        return [Ci_err, stdd_err, P_err]

    def gci_analysis(self, csv_file, starting_panel_size, panel_size_used):
        """
        Performs Grid Convergence Index (GCI) analysis on solver output data.

        Args:
            csv_file (str): Input CSV filename.
            starting_panel_size (int, optional): Minimum panel size to start GCI analysis.
            panel_size_used (int, optional): Panel size for which to report error/uncertainty.

        Prints:
            GCI results for lift and moment coefficients.
        """
        # Loads the output data from the solver
        df_out = pd.read_csv(self.dir + '/output/' + csv_file[:-4] + '_out.csv', delimiter=',')
        df_in = self.removed_failed_cases(csv_file)
        panel_indices = df_in.index[df_in['panel_size'] >= starting_panel_size].tolist()
        if panel_indices:
            starting_index = panel_indices[0]
        else:
            starting_index = 0  # fallback if not found
        panel_index = df_in.index[df_in['panel_size'] == panel_size_used].tolist()
        if panel_index:
            used_panel_index = panel_index
        else:
            used_panel_index = starting_index
            panel_size_used = starting_panel_size  # fallback if not found

        df_out["hs"] = mesh_size(1, df_in["panel_size"], 1)
        verify_second = Classic(df_out[starting_index:])
        verify_second.model.solve(p_limits=[0, 2])

        print("Lift coefficient limited order:              {:.6g}".format(verify_second.order['Cl']))
        print("Lift coefficient estimate:                   {:.6g}".format(verify_second.f_est['Cl']))
        # Extract scalar value for formatting
        cl_error = verify_second.error('Cl', used_panel_index)
        cl_uncertainty = verify_second.u('Cl', used_panel_index)
        if hasattr(cl_error, "__iter__"):
            cl_error = cl_error.iloc[0]
        if hasattr(cl_uncertainty, "__iter__"):
            cl_uncertainty = cl_uncertainty.iloc[0]
        print("Lift coefficient error for {} panels:       {:.6g}".format(panel_size_used, cl_error))
        print("Lift coefficient uncertainty for {} panels: {:.6g}".format(panel_size_used, cl_uncertainty))
        print()
        cm_error = verify_second.error('Cm', used_panel_index)
        cm_uncertainty = verify_second.u('Cm', used_panel_index)
        if hasattr(cm_error, "__iter__"):
            cm_error = cm_error.iloc[0]
        if hasattr(cm_uncertainty, "__iter__"):
            cm_uncertainty = cm_uncertainty.iloc[0]
        print("Moment coefficient limited order:              {:.6g}".format(verify_second.order['Cm']))
        print("Moment coefficient estimate:                   {:.6g}".format(verify_second.f_est['Cm']))
        print("Moment coefficient error for {} panels:       {:.6g}".format(panel_size_used, cm_error))
        print("Moment coefficient uncertainty for {} panels: {:.6g}".format(panel_size_used, cm_uncertainty))

    def surrogate_RMSE(self, csv_file_out, sur_file_out):
        """
        Calculates the Root Mean Square Error (RMSE) between surrogate model predictions and actual output data.

        Args:
            csv_file_out (str): Output sample CSV filename.
            sur_file_out (str): Output surrogate filename.

        Returns:
            np.ndarray: RMSE values for each output variable.
        """
        # Load input samples
        df_in = pd.read_csv(self.dir + '/output/' + csv_file_out, delimiter=',')
        # Load surrogate predictions
        df_sur = pd.read_csv(self.dir + '/output/' + sur_file_out, delimiter=',')
        # Ensure both DataFrames have the same number of rows
        if df_in.shape[0] != df_sur.shape[0]:
            raise ValueError("Input and surrogate DataFrames must have the same number of rows.")
        # Calculate RMSE for each output variable
        rmse = np.sqrt(np.mean((df_in.values - df_sur.values) ** 2, axis=0))
        return rmse
