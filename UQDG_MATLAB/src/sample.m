classdef sample < handle
    % Unified sampler for generating input samples for various distributions and sampling methods.
    %
    % Properties:
    %   Ns     - Number of samples
    %   solver - Solver type
    %   d      - Number of input variables
    %   fx     - Generated samples
    %   dir    - Working directory
    
    properties
        Ns         % Number of samples
        solver     % Solver type
        d          % Number of input variables
        fx         % Generated samples
        dir        % Working directory
        gci_panels % Panel sizes for GCI mode
    end
    
    methods
        function obj = sample(num_samples, solver_type)
            % Initialize the sample generator
            %
            % Args:
            %   num_samples (int, optional): Number of samples. Default is 50.
            %   solver_type (str, optional): Solver type. Default is 'xfoil'.
            
            if nargin < 1
                num_samples = 50;
            end
            if nargin < 2
                solver_type = 'xfoil';
            end
            
            obj.Ns = num_samples;
            obj.solver = solver_type;
            obj.d = [];
            obj.fx = [];
            obj.dir = pwd;
        end
        
        function create_samples(obj, csv_name, dist_type, method, param1, param2, varargin)
            % Create samples and save them to a CSV file
            %
            % Args:
            %   csv_name (str): Output CSV filename
            %   dist_type (str): 'normal', 'uniform', 'epistemic', or 'gci'
            %   method (str): 'sobol', 'lhs', 'monte', 'factorial'
            %   param1: For 'normal', mean array; for 'uniform' and 'epistemic', xmin array
            %   param2: For 'normal', std array; for 'uniform' and 'epistemic', xmax array
            %
            % Optional Name-Value pairs:
            %   'gci_panels' - Panel numbers for GCI study
            %   'input_names' - Cell array of input variable names
            %   'num_samples' - Override number of samples
            %   'solver' - Override solver type
            
            p = inputParser;
            addParameter(p, 'gci_panels', []);
            addParameter(p, 'input_names', []);
            addParameter(p, 'num_samples', []);
            addParameter(p, 'solver', []);
            parse(p, varargin{:});
            
            % Set sample size and solver if provided
            if ~isempty(p.Results.num_samples)
                obj.Ns = p.Results.num_samples;
            end
            if ~isempty(p.Results.solver)
                obj.solver = p.Results.solver;
            end
            
            % Handle GCI mode
            if strcmp(dist_type, 'gci')
                obj.d = length(param1);
                obj.fx = repmat(param1, obj.Ns, 1);
                obj.gci_panels = p.Results.gci_panels(:);
                obj.save_samples(csv_name, 'gci_mode', true, 'input_names', p.Results.input_names);
            else
                % Generate samples for other distributions
                obj.fx = obj.generate_samples(dist_type, method, param1, param2);
                obj.save_samples(csv_name, 'input_names', p.Results.input_names);
            end
        end
        
        function save_samples(obj, csv_name, varargin)
            % Save generated samples to a CSV file with two headers
            %
            % Args:
            %   csv_name (str): Output CSV filename
            %
            % Optional Name-Value pairs:
            %   'input_names' - Cell array of input variable names
            %   'gci_mode' - If true, append 'panel_size' column
            
            p = inputParser;
            addParameter(p, 'input_names', []);
            addParameter(p, 'gci_mode', false);
            parse(p, varargin{:});
            
            % Set default input names if not provided
            if isempty(p.Results.input_names)
                input_names = {'alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower'};
            else
                input_names = p.Results.input_names;
            end
            
            header1 = {obj.solver, obj.d, obj.Ns};
            header2 = input_names;
            
            % Add panel_size column for GCI mode
            if p.Results.gci_mode
                header2{end+1} = 'panel_size';
                data = [obj.fx, obj.gci_panels];
            else
                data = obj.fx;
            end
            
            out_path = fullfile(obj.dir, 'input', csv_name);
            
            % Write headers and data to CSV
            fid = fopen(out_path, 'w');
            fprintf(fid, '%s,%g,%g\n', header1{1}, header1{2}, header1{3});
            fprintf(fid, '%s', strjoin(header2, ','));
            fprintf(fid, '\n');
            fclose(fid);
            
            % Append data
            writematrix(data, out_path, 'WriteMode', 'append');
        end
        
        function fx = generate_samples(obj, dist_type, method, param1, param2)
            % Generate samples for a specified distribution and method
            %
            % Args:
            %   dist_type (str): 'normal', 'uniform', or 'epistemic'
            %   method (str): 'sobol', 'lhs', 'monte', or 'factorial'
            %   param1: For 'normal', mean array; for 'uniform' and 'epistemic', xmin array
            %   param2: For 'normal', std array; for 'uniform' and 'epistemic', xmax array
            %
            % Returns:
            %   fx: Generated samples
            
            % Set distribution parameters and dimension
            if strcmp(dist_type, 'normal')
                mean_val = param1(:)';
                std_val = param2(:)';
                obj.d = length(mean_val);
            elseif strcmp(dist_type, 'uniform')
                xmin = param1(:)';
                xmax = param2(:)';
                obj.d = length(xmin);
            elseif strcmp(dist_type, 'epistemic')
                xmin = param1(:)';
                xmax = param2(:)';
                obj.d = length(xmin);
                % Factorial grid for epistemic
                if strcmp(method, 'factorial')
                    n = round(obj.Ns^(1/obj.d));
                    ls = zeros(n, obj.d);
                    for i = 1:obj.d
                        ls(:, i) = linspace(xmin(i), xmax(i), n);
                    end
                    fx = zeros(n^obj.d, obj.d);
                    zz = 1;
                    % Nested loops for grid sampling (5D only)
                    for i = 1:n
                        for j = 1:n
                            for z = 1:n
                                for ii = 1:n
                                    for jj = 1:n
                                        fx(zz, :) = [ls(i,1), ls(j,2), ls(z,3), ls(ii,4), ls(jj,5)];
                                        zz = zz + 1;
                                    end
                                end
                            end
                        end
                    end
                    obj.fx = fx;
                    return;
                end
            end
            
            % Generate standard uniform samples
            if strcmp(method, 'sobol')
                su = sobolset(obj.d);
                su = net(su, obj.Ns);
            elseif strcmp(method, 'lhs')
                su = lhsdesign(obj.Ns, obj.d);
            elseif strcmp(method, 'monte')
                su = rand(obj.Ns, obj.d);
            end
            
            % Transform samples according to distribution
            if strcmp(dist_type, 'normal')
                fx = norminv(su);
                for i = 1:obj.d
                    fx(:, i) = std_val(i) * fx(:, i) + mean_val(i);
                end
                obj.fx = fx;
            elseif strcmp(dist_type, 'uniform')
                fx = zeros(obj.Ns, obj.d);
                for i = 1:obj.d
                    fx(:, i) = (xmax(i) - xmin(i)) * su(:, i) + xmin(i);
                end
                obj.fx = fx;
            end
        end
        
        function create_gci_samples(obj, csv_name, evaluation_point, panel_sizes, varargin)
            % Create and save samples for a GCI study
            %
            % Args:
            %   csv_name (str): Output CSV filename
            %   evaluation_point: Input parameter values
            %   panel_sizes: Panel size for each sample
            %
            % Optional Name-Value pairs:
            %   'input_names' - Cell array of input variable names
            %   'num_samples' - Override number of samples
            %   'solver' - Override solver type
            
            p = inputParser;
            addParameter(p, 'input_names', []);
            addParameter(p, 'num_samples', []);
            addParameter(p, 'solver', []);
            parse(p, varargin{:});
            
            % Set sample size and solver if provided
            if ~isempty(p.Results.num_samples)
                obj.Ns = p.Results.num_samples;
            else
                obj.Ns = length(panel_sizes);
            end
            if ~isempty(p.Results.solver)
                obj.solver = p.Results.solver;
            end
            
            obj.d = length(evaluation_point);
            if isempty(p.Results.input_names)
                input_names = {'alpha', 'Re', 'flap_deflection', 'xtr_upper', 'xtr_lower'};
            else
                input_names = p.Results.input_names;
            end
            
            % Create repeated input values and append panel sizes
            fx = repmat(evaluation_point(:)', obj.Ns, 1);
            panel_sizes = panel_sizes(:);
            header1 = {obj.solver, obj.d, obj.Ns};
            header2 = [input_names, {'panel_size'}];
            data = [fx, panel_sizes];
            
            out_path = fullfile(obj.dir, 'input', csv_name);
            
            % Write headers and data to CSV
            fid = fopen(out_path, 'w');
            fprintf(fid, '%s,%g,%g\n', header1{1}, header1{2}, header1{3});
            fprintf(fid, '%s', strjoin(header2, ','));
            fprintf(fid, '\n');
            fclose(fid);
            
            % Append data
            writematrix(data, out_path, 'WriteMode', 'append');
        end
        
        function mix_krig(obj, csv_name, xmin, xmax, varargin)
            % Generate a mixed training sample for kriging model
            %
            % Args:
            %   csv_name (str): Output CSV filename
            %   xmin: Minimum values for uniform distribution
            %   xmax: Maximum values for uniform distribution
            %
            % Optional Name-Value pairs:
            %   'epi_Ns' - Number of epistemic samples (default: 32)
            %   'input_names' - Input variable names for saving
            %   'num_samples' - Override number of samples
            %   'solver' - Override solver type
            
            p = inputParser;
            addParameter(p, 'epi_Ns', 32);
            addParameter(p, 'input_names', []);
            addParameter(p, 'num_samples', []);
            addParameter(p, 'solver', []);
            parse(p, varargin{:});
            
            % Set sample size and solver if provided
            if ~isempty(p.Results.num_samples)
                obj.Ns = p.Results.num_samples;
            end
            if ~isempty(p.Results.solver)
                obj.solver = p.Results.solver;
            end
            
            % Generate Sobol and epistemic samples
            obj.Ns = obj.Ns - p.Results.epi_Ns;
            sobol_fx = obj.generate_samples('uniform', 'sobol', xmin, xmax);
            obj.Ns = p.Results.epi_Ns;
            epi_fx = obj.generate_samples('epistemic', 'factorial', xmin, xmax);
            
            % Combine samples and save
            obj.fx = [sobol_fx; epi_fx];
            obj.Ns = size(obj.fx, 1);
            obj.d = size(obj.fx, 2);
            
            if ~isempty(p.Results.input_names)
                obj.save_samples(csv_name, 'input_names', p.Results.input_names);
            end
        end
    end
end
