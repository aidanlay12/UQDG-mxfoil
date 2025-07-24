classdef uq_analysis < handle
    % Performs uncertainty quantification (UQ) analysis on solver output data.
    %
    % This class provides methods for:
    % - Removing failed cases from input data.
    % - Calculating probability of outputs outside a specified P-box.
    % - Estimating uncertainty using Monte Carlo standard error.
    % - Computing RMSE between surrogate predictions and actual outputs.
    %
    % Note: For GCI (Grid Convergence Index) analysis, use the ORNL's cfd-verify Python library:
    %
    % Properties:
    %   dir    - Working directory
    %   nout   - Number of output variables
    %   Np     - Number of samples
    %   CMlim  - Moment coefficient limits for P-box
    %   CLlim  - Lift coefficient limits for P-box
    
    properties
        dir     % Working directory
        nout    % Number of output variables
        Np      % Number of samples
        CMlim   % Moment coefficient limits for P-box
        CLlim   % Lift coefficient limits for P-box
    end
    
    methods
        function obj = uq_analysis()
            % Initialize the uq_analysis object and its attributes
            obj.dir = pwd;
            obj.nout = 0;
            obj.Np = 0;
            obj.CMlim = [-0.05, -0.044];  % Moment coefficient range for P-box in challenge problem
            obj.CLlim = [0.15, 0.27];     % Lift coefficient range for P-box in challenge problem
        end
        
        function df = removed_failed_cases(obj, csv_file)
            % Reads the input data from the solver and removes failed cases
            %
            % Args:
            %   csv_file (str): Input CSV filename
            %
            % Returns:
            %   df: Input data with failed cases removed
            
            % Load the input data from the solver (skipping header row)
            input_path = fullfile(obj.dir, 'input', csv_file);
            opts = detectImportOptions(input_path);
            opts.DataLines = [3, Inf]; % Skip first two rows (headers)
            df = readtable(input_path, opts);
            
            fail_path = fullfile(obj.dir, 'output', [csv_file(1:end-4), '_fail.csv']);
            
            % Remove failed cases from input DataFrame if fail file exists
            if exist(fail_path, 'file')
                df_fail = readtable(fail_path);
                if ismember('Index', df_fail.Properties.VariableNames)
                    failed_indices = df_fail.Index + 1; % Convert to 1-based indexing
                    % Remove rows with these indices from df
                    df(failed_indices, :) = [];
                end
            end
        end
        
        function Pi = outside_pbox_prob(obj, data)
            % Calculates the percentage of samples outside the P-box bounds for lift and moment coefficients
            %
            % Args:
            %   data: Array of output samples (columns: cl, cm)
            %
            % Returns:
            %   Pi: Percentage of samples outside the P-box
            
            cl_out = (data(:, 1) < obj.CLlim(1)) | (data(:, 1) > obj.CLlim(2));
            cm_out = (data(:, 2) < obj.CMlim(1)) | (data(:, 2) > obj.CMlim(2));
            nout = sum(cl_out | cm_out);
            Pi = (nout / size(data, 1)) * 100;
        end
        
        function result = monte_carlo_SE(obj, csv_file, varargin)
            % Calculates uncertainty intervals for mean, standard deviation, and P-box probability using Monte Carlo method
            %
            % Args:
            %   csv_file (str): Input CSV filename
            %
            % Optional Name-Value pairs:
            %   'confidence_level_z' - Z-value for confidence interval (default: 1.96 for 95% CI)
            %
            % Returns:
            %   result: Cell array containing [mean uncertainty interval, std uncertainty interval, P-box uncertainty interval]
            
            p = inputParser;
            addParameter(p, 'confidence_level_z', 1.96);
            parse(p, varargin{:});
            
            % Load the output data from the solver
            out_path = fullfile(obj.dir, 'output', [csv_file(1:end-4), '_out.csv']);
            df = readtable(out_path);
            Np = height(df); % Number of samples
            z = p.Results.confidence_level_z; % 95% confidence interval constant
            
            Ci_err = zeros(1, 2);
            stdd_err = zeros(1, 2);
            
            % Calculate uncertainty intervals for mean and stddev for Cl and Cm
            var_names = {'Cl', 'Cm'};
            for idx = 1:2
                var_name = var_names{idx};
                data_col = df.(var_name);
                
                stddev = std(data_col, 1); % Using sample standard deviation
                conv = 1 / sqrt(Np);
                SE = stddev * conv;
                Ci_err(idx) = z * SE;
                
                mu = mean(data_col);
                mu4 = mean((data_col - mu).^4);
                SE = conv * sqrt((mu4 - (stddev^4) * (Np - 3) / (Np - 1)));
                SE = SE / (2 * stddev);
                stdd_err(idx) = z * SE;
            end
            
            % Calculate P-box probability and uncertainty
            out = table2array(df);
            Pmean = obj.outside_pbox_prob(out);
            Pstddev = sqrt(Pmean * (100 - Pmean));
            SE = Pstddev * conv;
            P_err = z * SE;
            
            result = {Ci_err, stdd_err, P_err};
        end
        
        function rmse = surrogate_RMSE(obj, csv_file_out, sur_file_out)
            % Calculates the Root Mean Square Error (RMSE) between surrogate model predictions and actual output data
            %
            % Args:
            %   csv_file_out (str): Output sample CSV filename
            %   sur_file_out (str): Output surrogate filename
            %
            % Returns:
            %   rmse: RMSE values for each output variable
            
            % Load input samples
            in_path = fullfile(obj.dir, 'output', csv_file_out);
            df_in = readtable(in_path);
            
            % Load surrogate predictions
            sur_path = fullfile(obj.dir, 'output', sur_file_out);
            df_sur = readtable(sur_path);
            
            % Ensure both DataFrames have the same number of rows
            if height(df_in) ~= height(df_sur)
                error('Input and surrogate DataFrames must have the same number of rows.');
            end
            
            % Calculate RMSE for each output variable
            in_data = table2array(df_in);
            sur_data = table2array(df_sur);
            rmse = sqrt(mean((in_data - sur_data).^2, 1));
        end
