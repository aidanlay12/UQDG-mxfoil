classdef krig_model < handle
    % Kriging surrogate model for spatial interpolation and uncertainty quantification.
    
    properties
        dir                 % Working directory
        Ns                  % Number of samples
        d                   % Input dimension
        std_sample          % Standardized input samples
        std_output          % Standardized output samples
        out_sample          % Raw output samples
        in_sample           % Raw input samples
        varmodel            % Fitted variogram parameters
        name_sample         % Name of input sample file
        dist                % Pairwise distances
    end
    
    methods
        function obj = krig_model()
            % Initialize the krig_model object and its attributes.
            obj.dir = pwd;
            obj.Ns = [];
            obj.d = [];
            obj.std_sample = [];
            obj.std_output = [];
            obj.out_sample = [];
            obj.in_sample = [];
            obj.varmodel = [];
            obj.name_sample = [];
            obj.dist = [];
        end
        
        function variogram(obj, out_var, num_bins)
            % Estimate variogram parameters for a given output variable.
            %
            % Args:
            %   out_var: Output variable index
            %   num_bins: Number of bins for distance grouping
            %
            % Sets:
            %   obj.varmodel: Fitted variogram parameters [Co, C, a]
            
            % Compute pairwise distances and semi-variances
            n = fix((obj.Ns*(obj.Ns-1))/2 + 1);
            obj.dist = zeros(n, 1);
            svar = zeros(n, 1);
            ii = 0;
            for i = 1:obj.Ns
                for j = (i+1):obj.Ns
                    ii = ii + 1;
                    obj.dist(ii) = norm(obj.std_sample(i,:) - obj.std_sample(j,:));
                    svar(ii) = 0.5*(obj.std_output(i,out_var) - obj.std_output(j,out_var))^2;
                end
            end
            
            % Bin distances and average semi-variances
            bins = linspace(0, max(obj.dist), num_bins+1);
            bin_centers = (bins(1:end-1) + bins(2:end)) / 2;
            svar_bin = zeros(num_bins, 1);
            for i = 1:num_bins
                mask = (obj.dist >= bins(i)) & (obj.dist < bins(i+1));
                if any(mask)
                    svar_bin(i) = mean(svar(mask));
                else
                    svar_bin(i) = NaN;
                end
            end
            
            % Remove invalid bins
            valid = ~isnan(svar_bin);
            h = bin_centers(valid);
            svar_valid = svar_bin(valid);
            
            % Ensure vectors are column vectors and same size
            h = h(:);
            svar_valid = svar_valid(:);
            
            % Fit Gaussian variogram model to binned semi-variances
            try
                % Initial parameter guess: [Co, C, a]
                p0 = [0.1, 1.0, 1.0];
                lb = [0, 0, 0];
                ub = [Inf, Inf, Inf];
                
                % Use lsqcurvefit for constrained fitting
                options = optimoptions('lsqcurvefit', 'Display', 'off', 'MaxFunctionEvaluations', 10000);
                fit_func = @(params, x) reshape(gaussian_variogram(x, params(1), params(2), params(3)), [], 1);
                popt = lsqcurvefit(fit_func, p0, h, svar_valid, lb, ub, options);
                obj.varmodel = popt;
            catch ME
                warning('krig_model:variogramFitting', 'Variogram fitting failed: %s. Using default parameters.', ME.message);
                obj.varmodel = [0.1, 1.0, 1.0];
            end
        end
        
        function assemble_surrogate(obj, training_data_file, sur_file, xmin, xmax, num_bins, gamma, max_gamma, cond_thresh)
            % Assemble and save the kriging surrogate model.
            %
            % Args:
            %   training_data_file: Input sample CSV filename
            %   sur_file: Output surrogate filename
            %   xmin: Minimum bounds for normalization
            %   xmax: Maximum bounds for normalization
            %   num_bins: Number of bins for variogram calculation (default 5)
            %   gamma: Initial regularization parameter (default 1e-6)
            %   max_gamma: Maximum regularization parameter (default 1e-2)
            %   cond_thresh: Condition number threshold (default 1e8)
            
            if nargin < 6; num_bins = 5; end
            if nargin < 7; gamma = 1e-6; end
            if nargin < 8; max_gamma = 1e-2; end
            if nargin < 9; cond_thresh = 1e8; end
            
            % Ensure xmin/xmax are row vectors
            xmin = xmin(:)';
            xmax = xmax(:)';
            
            % Standardize input/output data
            obj.std_out(training_data_file, xmin, xmax);
            
            for ii = 1:2
                % Fit variogram for each output variable
                obj.variogram(ii, num_bins);
                gamma = 1e-6;
                
                % Assemble variogram matrix
                A = zeros(obj.Ns + 1, obj.Ns + 1);
                for i = 1:obj.Ns
                    for j = 1:obj.Ns
                        h = norm(obj.std_sample(i, :) - obj.std_sample(j, :));
                        A(i, j) = gaussian_variogram(h, obj.varmodel(1), obj.varmodel(2), obj.varmodel(3));
                    end
                end
                A(obj.Ns+1, 1:obj.Ns) = ones(1, obj.Ns);
                A(1:obj.Ns, obj.Ns+1) = ones(obj.Ns, 1);
                A(end, end) = 0;
                
                % Adaptive regularization to ensure invertibility
                while true
                    A_reg = A + eye(size(A)) * gamma;
                    A_reg(end, end) = 0;
                    cond_num = cond(A_reg);
                    if cond_num < cond_thresh || gamma >= max_gamma
                        break;
                    end
                    gamma = gamma * 10;
                end
                Ainv = inv(A_reg);
                
                % Save standardized samples and outputs
                if ii == 1
                    % Create column headers
                    headers = {};
                    for i = 1:size(obj.std_sample, 2)
                        headers{end+1} = sprintf('std_sample_%d', i);
                    end
                    for i = 1:size(obj.std_output, 2)
                        headers{end+1} = sprintf('std_output_%d', i);
                    end
                    
                    % Write number of samples first
                    fid = fopen(fullfile(obj.dir, 'input', sur_file), 'w');
                    fprintf(fid, '%d\n', obj.Ns);
                    fclose(fid);
                    
                    % Append standardized data with headers
                    data_matrix = [obj.std_sample, obj.std_output];
                    T = array2table(data_matrix, 'VariableNames', headers);
                    writetable(T, fullfile(obj.dir, 'input', sur_file), 'WriteMode', 'append');
                    
                    % Append xmin and xmax without headers
                    writematrix(xmin, fullfile(obj.dir, 'input', sur_file), 'WriteMode', 'append');
                    writematrix(xmax, fullfile(obj.dir, 'input', sur_file), 'WriteMode', 'append');
                end
                
                % Save model parameters and Ainv
                out_mean = mean(obj.out_sample{:, ii});
                out_std = std(obj.out_sample{:, ii});
                params = [out_mean, out_std, obj.varmodel];
                
                % Create parameter table with headers
                param_headers = {'mean', 'std', 'Co', 'C', 'a'};
                T_params = array2table(params, 'VariableNames', param_headers);
                writetable(T_params, fullfile(obj.dir, 'input', sur_file), 'WriteMode', 'append');
                
                % Append Ainv matrix without headers
                writematrix(Ainv, fullfile(obj.dir, 'input', sur_file), 'WriteMode', 'append');
            end
        end
        
        function evaluate_surrogate(obj, csv_input, sur_file)
            % Evaluate the kriging surrogate model on new input data.
            %
            % Args:
            %   csv_input: Input sample CSV filename
            %   sur_file: Surrogate model filename
            %
            % Output:
            %   Writes predictions to output CSV with columns ['cl', 'cm']
            
            % Read number of surrogate samples
            fid = fopen(fullfile(obj.dir, 'input', sur_file), 'r');
            sur_Ns = str2double(fgetl(fid));
            fclose(fid);
            
            % Read standardized samples and outputs
            opts = detectImportOptions(fullfile(obj.dir, 'input', sur_file));
            opts.DataLines = [2, sur_Ns + 1]; % Skip first line (sample count)
            df_samples = readtable(fullfile(obj.dir, 'input', sur_file), opts);
            
            std_input = table2array(df_samples(:, 1:obj.d));
            std_output = table2array(df_samples(:, obj.d+1:obj.d+2));
            
            % Read xmin and xmax
            norm_start = 2 + sur_Ns;
            opts_norm = detectImportOptions(fullfile(obj.dir, 'input', sur_file));
            opts_norm.DataLines = [norm_start, norm_start + 1];
            opts_norm.VariableNamesLine = 1; % No headers
            df_norm = readmatrix(fullfile(obj.dir, 'input', sur_file), 'Range', [norm_start, 1, norm_start + 1, obj.d]);
            xmin = df_norm(1, :);
            xmax = df_norm(2, :);
            
            % Standardize new input samples
            obj.std_out(csv_input, xmin, xmax);
            
            sout = zeros(obj.Ns, 2); % Output array for predictions
            
            % Loop over each output variable
            current_row = 4 + sur_Ns;
            for ii = 1:2
                % Read model parameters
                opts_params = detectImportOptions(fullfile(obj.dir, 'input', sur_file));
                opts_params.DataLines = [current_row, current_row];
                opts_params.VariableNamesLine = 1; % No headers for this read
                params_data = readmatrix(fullfile(obj.dir, 'input', sur_file), 'Range', [current_row, 1, current_row, 5]);
                
                out_mean = params_data(1);
                out_stddev = params_data(2);
                varmodel = params_data(3:5);
                current_row = current_row + 1;
                
                % Read Ainv matrix
                Ainv = readmatrix(fullfile(obj.dir, 'input', sur_file), ...
                    'Range', [current_row, 1, current_row + sur_Ns, sur_Ns + 1]);
                current_row = current_row + sur_Ns + 1;  % Fixed: was + 2, should be + 1
                
                % Compute distances between test points and surrogate input points
                h = zeros(obj.Ns, sur_Ns);
                for i = 1:obj.Ns
                    for j = 1:sur_Ns
                        h(i, j) = norm(obj.std_sample(i, :) - std_input(j, :));
                    end
                end
                
                % Compute variogram values
                b = gaussian_variogram(h, varmodel(1), varmodel(2), varmodel(3));
                
                % Build kriging weights - b is (Ns x sur_Ns), create full matrix with ones column
                b_full = ones(obj.Ns, sur_Ns + 1);
                b_full(:, 1:sur_Ns) = b;
                weights = (Ainv * b_full')';
                
                % Compute surrogate predictions
                z = weights(:, 1:sur_Ns) * std_output(:, ii);
                sout(:, ii) = z * out_stddev + out_mean;
            end
            
            % Save predictions to output CSV
            output_table = array2table(sout, 'VariableNames', {'cl', 'cm'});
            output_filename = fullfile(obj.dir, 'output', [obj.name_sample(1:end-4), '_', sur_file(1:end-4), '_sout.csv']);
            writetable(output_table, output_filename);
        end
        
        function std_out(obj, in_sample, xmin, xmax)
            % Load and standardize input/output data.
            %
            % Args:
            %   in_sample: Input sample CSV filename
            %   xmin: Minimum bounds for normalization
            %   xmax: Maximum bounds for normalization
            %
            % Sets:
            %   obj.std_sample: Standardized input samples
            %   obj.std_output: Standardized output samples
            %   obj.in_sample: Cleaned input samples
            %   obj.out_sample: Raw output samples
            %   obj.Ns: Number of samples
            %   obj.d: Input dimension
            
            % Ensure xmin/xmax are row vectors
            xmin = xmin(:)';
            xmax = xmax(:)';
            
            obj.name_sample = in_sample;
            out_path = fullfile(obj.dir, 'output', [obj.name_sample(1:end-4), '_out.csv']);
            obj.out_sample = readtable(out_path);
            
            % Remove failed samples
            uqa = uq_analysis();
            obj.in_sample = uqa.removed_failed_cases(obj.name_sample);
            obj.Ns = size(obj.in_sample, 1);
            obj.d = size(obj.in_sample, 2);
            
            % Standardize input
            obj.std_sample = (table2array(obj.in_sample) - xmin) ./ (xmax - xmin);
            
            % Standardize output
            obj.std_output = zeros(size(obj.out_sample, 1), size(obj.out_sample, 2));
            for i = 1:size(obj.out_sample, 2)
                col = obj.out_sample{:, i};
                col_std = std(col);
                col_mean = mean(col);
                obj.std_output(:, i) = (col - col_mean) / col_std;
            end
        end
    end
end

function var = gaussian_variogram(magh, Co, C, a)
    % Gaussian variogram model.
    %
    % Args:
    %   magh: Distance(s) between points
    %   Co: Nugget parameter
    %   C: Sill parameter
    %   a: Range parameter
    %
    % Returns:
    %   var: Variogram value(s) - preserves input dimensions
    
    var = Co + C * (1 - exp(-(magh / a).^2));
    % Don't force column vector - preserve original dimensions
end
