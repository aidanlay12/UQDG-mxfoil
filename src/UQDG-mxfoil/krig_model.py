import numpy as np
import pandas as pd
import os
import math
import ast
from UQDG-mxfoil.uq_analysis import *

def gaussian_variogram(magh,Co,C,a):
    #Uses an gaussian model for the variogram
    var = Co + C * (1 - np.exp(-(magh / a)**2))

    return var

class krig_model():
    def __init__(self):
        self.dir = os.getcwd()
        pass

    def variogram(self,out_var,num_bins):
        #Tries to find the variogram coefficient (NOT SURE IF IT 100% correct)


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
        popt, pconv = curve_fit(
            gaussian_variogram,
            h,
            svar,
            p0=[0.1, 1.0, 1.0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),  # All parameters >= 0
            maxfev=10000
        )

        self.varmodel = popt #Saves the model parameters


    def assem(self, sur_file, xmin=None, xmax=None):
        # Loads the output data and writes all results using pandas
        for ii in range(2):
            self.variogram(ii, 5)  # Creates a variogram
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
            gamma = 1e-8
            max_gamma = 1e-2
            cond_thresh = 1e8
            while True:
                A_reg = A + np.eye(A.shape[0]) * gamma
                A_reg[-1, -1] = 0
                cond = np.linalg.cond(A_reg)
                if cond < cond_thresh or gamma >= max_gamma:
                    break
                gamma *= 10
            print(gamma)
            Ainv = np.linalg.inv(A_reg)

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
                df_samples.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)

                # Save normalization bounds as a separate block
                if xmin is not None and xmax is not None:
                    df_norm = pd.DataFrame({'xmin': [list(xmin)], 'xmax': [list(xmax)]})
                    df_norm.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)

            # Second block: model parameters and Ainv
            params = [self.out_sample[:, ii].mean(), self.out_sample[:, ii].std(), self.varmodel[0], self.varmodel[1], self.varmodel[2]]
            df_params = pd.DataFrame([params], columns=['mean', 'std', 'Co', 'C', 'a'])
            df_params.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False)

            df_Ainv = pd.DataFrame(Ainv)
            df_Ainv.to_csv(self.dir + '/input/' + sur_file, mode='a', index=False, header=False)
    

    def eval(self, sur_file):
        # Loads the input data using pandas
        num_tests = self.Ns

        # Read the number of samples (first line)
        with open(self.dir + '/input/' + sur_file, 'r') as f:
            sur_Ns = int(f.readline().strip())

        # Read the sample data block
        df_samples = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=1, nrows=sur_Ns)
        std_input = df_samples.iloc[:, :self.d].values
        std_output = df_samples.iloc[:, self.d:].values

        sout = np.zeros((num_tests, 2))  # Output array for predictions

        # Track the current row for reading parameters and Ainv
        current_row = 1 + sur_Ns
        for ii in range(2):
            # Read model parameters
            df_params = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=current_row, nrows=1)
            out_mean = df_params.iloc[0, 0]
            out_stddev = df_params.iloc[0, 1]
            varmodel = df_params.iloc[0, 2:].values

            current_row += 1
            # Read Ainv
            df_Ainv = pd.read_csv(self.dir + '/input/' + sur_file, skiprows=current_row, nrows=sur_Ns + 1, header=None)
            Ainv = df_Ainv.values
            current_row += sur_Ns + 1

            # Vectorized distance computation between all test points and sur_input points
            diff = self.std_sample[:, None, :] - std_input[None, :, :]
            h = np.linalg.norm(diff, axis=2)
            b = gaussian_variogram(h, varmodel[0], varmodel[1], varmodel[2])

            b_full = np.ones((num_tests, sur_Ns + 1))
            b_full[:, :-1] = b

            weights = Ainv @ b_full.T
            weights = weights.T

            z = np.dot(weights[:, :sur_Ns], std_output[:, ii])
            sout[:, ii] = z * out_stddev + out_mean

        # Save predictions using pandas
        df_sout = pd.DataFrame(sout)
        df_sout.to_csv(self.dir + '/output/' + self.name_sample[:-4] + '_' + sur_file +'_sout.csv', index=False, header=False)


    def std_out(self, in_sample, xmin, xmax):
        # Loads and standardizes input/output data using pandas, expects headers in CSVs
        self.name_sample = in_sample
        self.out_sample = pd.read_csv(self.dir + "/output/" + self.name_sample[:-4] + "_out.csv", delimiter=',')

        # Remove failed samples
        uqa = uq_analysis()
        self.in_sample = uqa.removed_failed_cases(self.name_sample)

        self.Ns = self.in_sample.shape[0]
        self.d = self.in_sample.shape[1]

        # Standardize input
        self.std_sample = (self.in_sample.values - np.array(xmin)) / (np.array(xmax) - np.array(xmin))

        # Standardize output
        self.std_output = np.zeros(self.out_sample.shape)
        for i in range(self.out_sample.shape[1]):
            col = self.out_sample.iloc[:, i]
            self.std_output[:, i] = (col - col.mean()) / col.std()