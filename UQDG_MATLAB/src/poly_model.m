classdef poly_model < handle
    % Polynomial Chaos Surrogate Model
    %
    % This class provides methods to assemble and evaluate polynomial chaos surrogates for aerodynamic analysis.
    % It supports normalization of input data, robust coefficient extraction, and integration with UQ analysis.
    %
    % Key Features:
    % - One-line surrogate assembly and evaluation
    % - Normalization bounds (xmin, xmax) saved and loaded with coefficients
    % - Uses tables for reliable input/output
    % - Removes failed samples using UQ analysis
    % - Vectorized evaluation of polynomial basis for all samples
    
    properties
        dir          % Working directory
        n            % Number of input variables
        p            % Polynomial degree
        npb          % Number of polynomial basis functions
        Ns           % Number of samples
        std_sample   % Standardized input samples
        in_sample    % Raw input samples
        out_sample   % Output samples
        name_sample  % Name of input sample file
    end
    
    methods
        function obj = poly_model()
            % Initialize the poly_model object
            obj.dir = pwd;
        end
        
        function b = basis(obj, x, d)
            % Legendre polynomial basis up to degree 2
            %
            % Args:
            %   x: Input value
            %   d: Polynomial degree
            %
            % Returns:
            %   b: Basis function value
            
            switch d
                case 0
                    b = 1;
                case 1
                    b = x;
                case 2
                    b = 0.5 * (3 * x^2 - 1);
                otherwise
                    error('Polynomial degree %d not implemented', d);
            end
        end
        
        function pb = poly_basis(obj, in_sample)
            % Generates the polynomial basis vector for a given standardized input sample
            %
            % Args:
            %   in_sample: Standardized input sample (row vector)
            %
            % Returns:
            %   pb: Vector of polynomial basis coefficients
            
            % Ensure input is row vector
            in_sample = in_sample(:)';
            pb = zeros(obj.npb, 1);
            pb(1) = 1; % Constant term
            
            % Non-cross terms: each variable, each polynomial degree
            idx = 2;
            for degree = 1:obj.p
                for var = 1:obj.n
                    pb(idx) = obj.basis(in_sample(var), degree);
                    idx = idx + 1;
                end
            end
            
            % Cross-terms (only for p > 1): products of basis functions for variable pairs
            if obj.p > 1
                for i = 1:obj.n
                    for j = (i+1):obj.n
                        pb(idx) = obj.basis(in_sample(i), 1) * obj.basis(in_sample(j), 1);
                        idx = idx + 1;
                    end
                end
            end
        end
        
        function assemble_surrogate(obj, csv_input, polynomial_degree, xmin, xmax, poly_coefficients_csv)
            % Assembles the polynomial chaos surrogate and saves coefficients and normalization bounds in one step
            %
            % Args:
            %   csv_input (str): Input sample CSV filename
            %   polynomial_degree (int): Degree of polynomial chaos interpolation
            %   xmin: Minimum values for each variable for normalization
            %   xmax: Maximum values for each variable for normalization
            %   poly_coefficients_csv (str): Output CSV filename for coefficients
            
            % Ensure xmin/xmax are row vectors
            xmin = xmin(:)';
            xmax = xmax(:)';
            
            % Read input sample and determine n
            input_path = fullfile(obj.dir, 'input', csv_input);
            opts = detectImportOptions(input_path);
            opts.DataLines = [3, Inf]; % Skip first two rows (headers)
            sample_df = readtable(input_path, opts);
            obj.n = width(sample_df);
            obj.p = polynomial_degree;
            obj.npb = factorial(obj.n + obj.p) / (factorial(obj.n) * factorial(obj.p));
            
            % Standardize and assemble
            obj.std_out(csv_input, xmin, xmax);
            obj.assem(poly_coefficients_csv, xmin, xmax);
        end
        
        function assem(obj, poly_coefficients_csv, xmin, xmax)
            % Assembles polynomial coefficients from standardized input and output data, and saves them to CSV
            % Also saves normalization bounds as columns in the output CSV
            %
            % Args:
            %   poly_coefficients_csv (str): Output CSV filename for coefficients
            %   xmin: Minimum values for normalization
            %   xmax: Maximum values for normalization
            
            % Read output sample
            out_path = fullfile(obj.dir, 'output', [obj.name_sample(1:end-4), '_out.csv']);
            obj.out_sample = readtable(out_path);
            Nsol = width(obj.out_sample);
            output_header = obj.out_sample.Properties.VariableNames;
            sol = zeros(obj.npb, Nsol);
            
            % Loop over each output variable
            for j = 1:Nsol
                A = zeros(obj.Ns, obj.npb);
                b = obj.out_sample{:, j};
                
                % Build design matrix for least squares
                for i = 1:obj.Ns
                    A(i, :) = obj.poly_basis(obj.std_sample(i, :))';
                end
                
                % Solve least squares for coefficients
                sol(:, j) = A \ b;
            end
            
            % Prepare header for output CSV
            if Nsol == 2
                header = {'Cl_model', 'Cm_model'};
            else
                header = cell(1, Nsol);
                for i = 1:Nsol
                    header{i} = sprintf('out_model_%d', i);
                end
            end
            
            out_path_coeff = fullfile(obj.dir, 'input', poly_coefficients_csv);
            
            % Write normalization bounds and coefficients to CSV
            fid = fopen(out_path_coeff, 'w');
            fprintf(fid, '%s\n', strjoin(string(xmin), ','));
            fprintf(fid, '%s\n', strjoin(string(xmax), ','));
            fclose(fid);
            
            % Append coefficients
            coeff_table = array2table(sol, 'VariableNames', header);
            writetable(coeff_table, out_path_coeff, 'WriteMode', 'append');
        end
        
        function std_out(obj, in_sample_csv, xmin, xmax)
            % Loads and standardizes input data using provided normalization bounds
            % Removes failed samples using UQ analysis
            %
            % Args:
            %   in_sample_csv (str): Input sample CSV filename
            %   xmin: Minimum values for normalization
            %   xmax: Maximum values for normalization
            
            % Ensure xmin/xmax are row vectors
            xmin = xmin(:)';
            xmax = xmax(:)';
            
            obj.name_sample = in_sample_csv;
            
            % Remove failed cases using UQ analysis
            uqa = uq_analysis();
            obj.in_sample = uqa.removed_failed_cases(obj.name_sample);
            obj.Ns = height(obj.in_sample);
            obj.std_sample = zeros(obj.Ns, width(obj.in_sample));
            
            % Standardize each column
            in_data = table2array(obj.in_sample);
            for i = 1:width(obj.in_sample)
                obj.std_sample(:, i) = (in_data(:, i) - xmin(i)) / (xmax(i) - xmin(i));
            end
        end
        
        function evaluate_surrogate(obj, csv_input, poly_c_csv)
            % Evaluates the polynomial chaos surrogate using coefficients and normalization bounds from CSV
            % Standardizes the input data and computes surrogate predictions for all samples
            %
            % Args:
            %   csv_input (str): Input sample CSV filename
            %   poly_c_csv (str): CSV file with surrogate coefficients and normalization bounds
            
            coeff_path = fullfile(obj.dir, 'input', poly_c_csv);
            
            % Read normalization bounds from first two rows of coefficients CSV
            fid = fopen(coeff_path, 'r');
            xmin_line = fgetl(fid);
            xmax_line = fgetl(fid);
            fclose(fid);
            
            xmin_parts = strsplit(xmin_line, ',');
            xmax_parts = strsplit(xmax_line, ',');
            xmin = str2double(xmin_parts);
            xmax = str2double(xmax_parts);
            
            % Read coefficients (skip first two rows)
            opts = detectImportOptions(coeff_path);
            opts.DataLines = [3, Inf];
            pc = readtable(coeff_path, opts);
            header = pc.Properties.VariableNames;
            pc_matrix = table2array(pc);
            
            % Determine polynomial parameters from coefficient matrix size
            obj.npb = size(pc_matrix, 1);
            obj.n = length(xmin);
            
            % Estimate polynomial degree (simplified approach)
            % For small problems, we can solve: npb = factorial(n+p)/(factorial(n)*factorial(p))
            % This is a simplified estimation
            for p_test = 1:5
                npb_test = factorial(obj.n + p_test) / (factorial(obj.n) * factorial(p_test));
                if npb_test == obj.npb
                    obj.p = p_test;
                    break;
                end
            end
            
            % Standardize input samples
            obj.std_out(csv_input, xmin, xmax);
            
            % Build polynomial basis for all samples
            PB = zeros(obj.Ns, obj.npb);
            for i = 1:obj.Ns
                PB(i, :) = obj.poly_basis(obj.std_sample(i, :))';
            end
            
            % Surrogate prediction
            sol = PB * pc_matrix;
            
            % Write predictions to output CSV
            output_csv = fullfile(obj.dir, 'output', ...
                [obj.name_sample(1:end-4), '_', poly_c_csv(1:end-4), '_sout.csv']);
            
            sol_table = array2table(sol, 'VariableNames', header);
            writetable(sol_table, output_csv);
        end
    end
end
