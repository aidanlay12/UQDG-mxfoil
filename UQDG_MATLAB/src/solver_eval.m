classdef solver_eval < handle
    % Base class for running aerodynamic analyses using mfoil (MATLAB version).
    %
    % This class reads a double-header CSV input file, runs mfoil for each sample,
    % and writes results to output files. Supports UQ, mesh convergence (GCI), and solver verification.
    %
    % Input CSV format:
    %   - First row: solver type, number of variables, number of samples (e.g. mfoil,5,100)
    %   - Second row: variable names (e.g. alpha,Re,flap_deflection,xtr_lower,xtr_upper,...)
    %   - Data rows: values for each sample
    %
    % Output files:
    %   - <input_file>_out.csv: Results for each sample (columns: Cl, Cm)
    %   - <input_file>_fail.csv: Failed cases (index + input parameters)
    %
    % Usage:
    %   solver = solver_eval('input.csv');
    %   solver.run();  % Runs all cases
    %   solver.single_solve(i);  % Runs a single case (i-th sample)
    
    properties
        dir           % Current working directory
        solver        % Solver type (always mfoil for MATLAB version)
        d             % Number of independent variables
        Ns            % Total number of samples or runs
        results       % Results of the mfoil solver
        airfoil_name  % Airfoil name for mfoil
        panel_size    % Number of panels for mfoil
        conv_tol      % Convergence criteria for mfoil
        df            % Data table from CSV input
        input_names   % Input variable names from the DataFrame
        input_file    % Input CSV filename (without path)
    end
    
    methods
        function obj = solver_eval(inputfile, varargin)
            % Initialize the solver (MATLAB version uses mfoil only)
            %
            % Args:
            %   inputfile (str): CSV file with double header and sample data
            %
            % Optional Name-Value pairs:
            %   'panel_size' - Default number of panels for mfoil (default: 256)
            %   'conv_tol' - Convergence tolerance for mfoil (default: 1e-4)
            %   'airfoil_name' - Airfoil name (default: 'NACA 2412')
            
            p = inputParser;
            addParameter(p, 'panel_size', 256);
            addParameter(p, 'conv_tol', 1e-4);
            addParameter(p, 'airfoil_name', 'NACA 2412');
            parse(p, varargin{:});
            
            obj.dir = pwd;
            obj.airfoil_name = p.Results.airfoil_name;
            obj.panel_size = p.Results.panel_size;
            obj.conv_tol = p.Results.conv_tol;
            obj.input_file = inputfile;  % Store input filename
            
            % Read double header
            csv_path = fullfile(obj.dir, 'input', inputfile);
            
            % Read first header line
            fid = fopen(csv_path, 'r');
            first_header = strsplit(fgetl(fid), ',');
            fclose(fid);
            
            % Force mfoil solver for MATLAB version (ignore CSV header)
            obj.solver = 'mfoil';
            obj.d = str2double(first_header{2});
            obj.Ns = str2double(first_header{3});
            fprintf('MATLAB version: Forced to use mfoil solver\n');
            
            % Read CSV data (skipping first row)
            opts = detectImportOptions(csv_path);
            opts.DataLines = [3, Inf]; % Skip first two rows
            obj.df = readtable(csv_path, opts);
            obj.input_names = obj.df.Properties.VariableNames;
        end
        
        function run(obj)
            % Runs all cases in the input file (calls multi_eval)
            % Usage: solver_eval('input.csv').run()
            obj.multi_eval();
        end
        
        function single_solve(obj, input_index, varargin)
            % Runs the solver for a single sample (MATLAB version only uses mfoil)
            %
            % Args:
            %   input_index (int): Index of the sample in the DataFrame
            %
            % Optional Name-Value pairs:
            %   'panel_size' - Override panel size for this run
            
            p = inputParser;
            addParameter(p, 'panel_size', []);
            parse(p, varargin{:});
            
            % Check if 'panel_size' column exists in input CSV and use its value if present
            if ismember('panel_size', obj.input_names)
                panel = obj.df.panel_size(input_index);
            else
                panel = p.Results.panel_size;
                if isempty(panel)
                    panel = obj.panel_size;
                end
            end
            
            % MATLAB version only uses mfoil
            obj.msolve(input_index, panel);
        end
        
        function msolve(obj, input_index, panel)
            % Runs mfoil for a single case
            % Sets solver parameters, runs, and saves results to obj.results
            %
            % Args:
            %   input_index (int): Index of the sample in the DataFrame
            %   panel (int): Number of panels for mfoil
            
            % Get input data for this sample
            fx = table2array(obj.df(input_index, :));
            
            % Extract NACA airfoil number from airfoil name (e.g., 'NACA 2412' -> '2412')
            naca_str = regexprep(obj.airfoil_name, '[^0-9]', '');
            
            % Initialize mfoil with NACA airfoil coordinates
            msolve = mfoil('naca', naca_str, 'npanel', panel);
            
            % Set the flap angle
            msolve.geom_flap([0.7, 0.015], fx(3));
            
            % Set the operating conditions using the input data
            msolve.setoper('alpha', fx(1), 'Re', fx(2), 'Ma', 0, 'visc', true, ...
                          'xftu', fx(4), 'xftl', fx(5));
            
            % Configure solver parameters
            msolve.param.doplot = false;
            msolve.param.rtol = obj.conv_tol;
            msolve.param.verb = 0;
            
            % Try to solve mfoil and if it fails write the results as 0s
            try
                msolve.solve();
                
                if msolve.glob.conv == true
                    obj.results = [msolve.post.cl, msolve.post.cm];
                else
                    obj.results = [0.0, 0.0];
                end
                
            catch
                obj.results = [0.0, 0.0];
            end
        end
        
        function multi_eval(obj)
            % Runs the solver for all samples in the input file.
            % For each sample:
            %   - Calls single_solve()
            %   - Appends results to output CSV
            %   - Logs failed cases to fail CSV
            %   - Prints progress and estimated time remaining
            
            start_time = tic;
            out_path = fullfile(obj.dir, 'output', [obj.input_file(1:end-4), '_out.csv']);
            fail_path = fullfile(obj.dir, 'output', [obj.input_file(1:end-4), '_fail.csv']);
            
            % Write header if file does not exist
            if ~exist(out_path, 'file')
                writecell({'Cl', 'Cm'}, out_path);
            end
            
            % Loop through all samples
            for i = 1:obj.Ns
                obj.single_solve(i);
                
                if obj.results(1) ~= 0
                    writematrix(obj.results, out_path, 'WriteMode', 'append');
                else
                    % Write fail file only if a failed case occurs
                    fail_columns = [{'Index'}, obj.input_names];
                    if ~exist(fail_path, 'file')
                        writecell(fail_columns, fail_path);
                    end
                    fail_data = [i, table2array(obj.df(i, :))];
                    writematrix(fail_data, fail_path, 'WriteMode', 'append');
                    fprintf('Failed\n');
                end
                
                % Print progress and estimated time remaining
                elapsed = toc(start_time);
                finish_time = (elapsed / i) * (obj.Ns - i);
                hours = floor(finish_time / 3600);
                minutes = floor((finish_time - hours * 3600) / 60);
                seconds = floor(finish_time - hours * 3600 - minutes * 60);
                fprintf('Time: %02d:%02d:%02d Progress: %d/%d\n', ...
                    hours, minutes, seconds, i, obj.Ns);
            end
        end
    end
end
