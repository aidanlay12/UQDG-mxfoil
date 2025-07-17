
class uq_analysis:
    """
    This class performs uncertainty quantification analysis on the output data from a solver.
    It calculates the uncertainty of the output data using the Monte Carlo method.
    """

    def __init__(self, dir):
        self.dir = dir
        self.nout = 0
        self.Np = 0
        self.CMlim = np.float64([-0.05,-0.044]) #Moment coefficient range for P-box in challenge problem
        self.CLlim = np.float64([0.15,0.27]) #Lift coefficient range for P-box in challenge problem

    def removed_failed_cases(self, csv_file):
        """
        Reads the input data from the solver and removes the failed cases from the input data.
        The failed cases are stored in a separate file.
        """
        # Loads the input data from the solver
        df = pd.read_csv(self.dir+'/input/'+csv_file, delimiter=',')    

        fail_path = self.dir+'/output/'+csv_file[:-4]+'_fail.csv'
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
        # Vectorized calculation of the sample P-box probability
        cl_out = (data[:, 0] < self.CLlim[0]) | (data[:, 0] > self.CLlim[1])
        cm_out = (data[:, 1] < self.CMlim[0]) | (data[:, 1] > self.CMlim[1])
        nout = np.count_nonzero(cl_out | cm_out)
        Pi = (nout / data.shape[0])*100
        return Pi

    def monte_carlo_SE(self,csv_file):
        # Calculates the uncertainty of the output data from the solver using the Monte Carlo method.
        # Reads the output data from the solver and calculates the standard error and uncertainty interval for the mean, standard deviation, and P-box probability.
        
        # Loads the output data from the solver
        df = pd.read_csv(self.dir+'/output/'+csv_file[:-4]+'_out.csv', delimiter=',')    

        #Extracts the output data from the dataframe but leaves out the headers
        Np = len(df)  # Number of samples

        # Initializes the constant for 95% confidence interval
        z = 1.96 
        
        #Initialzes the array for the uncertianty interval and standard devaitation for the mean
        Ci_err = np.zeros(2)
        stdd_err = np.zeros(2)
    
        #Calculates the standard error for the mean and standard deviation for a 95% confidence interval
        for i in ['Cl','Cm']:
            #Calculates the standard deviation of the output data
            stddev = np.std(df[i], ddof=1)

            #Calculates the standard error for the mean
            conv = 1/np.sqrt(Np)

            conv = np.log(Np)**5/Np
            SE = stddev*conv

            #Calculates the uncertainty interval for the mean for a 95% confidence interval
            Ci_err[i] = z*SE
            

            #Calculates the mean of the output data
            mu = np.mean(df[i])

            #Calcualtes the fourth moment of the output data
            mu4 = np.mean((df[i] - mu)**4)

            #Calcualtes the standard error for the variance
            SE = conv*np.sqrt((mu4 - (stddev**4)*(Np-3)/(Np-1)))

            #Calculates the standard error for the standard deviation
            SE = SE/(2*stddev)

            #Calcuates the uncertainty interval for the standard deviation for a 95% confidence interval
            stdd_err[i] = z*SE
            
        #Calculates the P-box probability for the output data
        out = df.to_numpy()[:,1:]
        Pmean = self.outside_pbox_prob(out)

        #Calculates the standard deviation for the P-box probability using the binomial distribution
        Pstddev = np.sqrt(Pmean*(100-Pmean))

        #Calculates the standard error for the P-box probability
        SE = Pstddev*conv
        
        #Calculates the uncertainty interval for the P-box probability for a 95% confidence interval
        P_err = z*SE
            
        return [Ci_err,stdd_err,P_err]
    
    def gci_analysis(self, csv_file, starting_panel_size=200, panel_size_used=256):
        """
        Performs GCI analysis on the output data from the solver.
        Reads the output data from the solver and calculates the GCI for the lift and moment coefficients.
        """
        # Loads the output data from the solver
        df_out = pd.read_csv(self.dir+'/output/'+csv_file[:-4]+'_out.csv', delimiter=',') 
        df_in = self.removed_failed_cases(csv_file)

        panel_indices = df_in.index[df_in['panels'] >= starting_panel_size].tolist()
        if panel_indices:
            starting_index = panel_indices[0]
        else:
            starting_index = 0  # fallback if not found

        panel_index = df_in.index[df_in['panels'] == desired_panel_size].tolist()
        if panel_index:
            used_panel_index = panel_index[0]
        else:
            used_panel_index = starting_index
            panel_size_used = starting_panel_size  # fallback if not found

        verify_second.model.solve(p_limits=[0,2])
        verify_first = Classic(data.drop("panels", axis=1)[:starting_index])
        verify_second = Classic(data.drop("panels", axis=1)[starting_index:])

        print(f"Lift coefficient limited order:              {verify_second.order['cl']:.6g}")
        print(f"Lift coefficient estimate:                   {verify_second.f_est['cl']:.6g}")
        print(f"Lift coefficient error for {panel_size_used} panels:       {verify_second.error('cl', used_panel_index):.6g}")
        print(f"Lift coefficient uncertainty for {panel_size_used} panels: {verify_second.u('cl',used_panel_index):.6g}")
        print()

        print(f"Moment coefficient limited order:              {verify_second.order['cm']:.6g}")
        print(f"Moment coefficient estimate:                   {verify_second.f_est['cm']:.6g}")
        print(f"Moment coefficient error for {panel_size_used} panels:       {verify_second.error('cm', used_panel_index):.6g}")
        print(f"Moment coefficient uncertainty for {panel_size_used} panels: {verify_second.u('cm',used_panel_index):.6g}")
