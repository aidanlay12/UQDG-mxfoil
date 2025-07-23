import os
import sys
import re
import numpy as np
import pandas as pd
from subprocess import Popen, TimeoutExpired, DEVNULL
from time import monotonic as timer
sys.path.append(os.path.join(os.getcwd(), "src/solvers"))
from mfoil import *



def check_line_data(file_path, line_number):
    """Check if a specific line in a text file has data."""
    #Given a text file (file_path), this function can determine whether a line in the text file
    #has any data. This is important for when xfoil does not converge and does not output anydata
    with open(file_path, 'r') as file:
        for i, line in enumerate(file): #Scans throught the file to check the lines
            if i + 1 == line_number:
                return bool(line.strip())  # Return True if line is not empty after removing whitespace
        return False  # Line number exceeds the number of lines in the file


class solver_eval:
    """
    Base class for running aerodynamic analyses using xfoil or mfoil.

    This class reads a double-header CSV input file, runs the specified solver for each sample,
    and writes results to output files. Supports UQ, mesh convergence (GCI), and solver verification.

    Input CSV format:
        - First row: solver type, number of variables, number of samples (e.g. xfoil,5,100)
        - Second row: variable names (e.g. alpha,Re,flap_deflection,xtr_lower,xtr_upper,...)
        - Data rows: values for each sample

    Output files:
        - <input_file>_out.csv: Results for each sample (columns: Cl, Cm)
        - <input_file>_fail.csv: Failed cases (index + input parameters)

    Usage:
        solver = solver_eval('input.csv')
        solver.run()  # Runs all cases
        solver.single_solve(i)  # Runs a single case (i-th sample)
        solver.single_solve(i, panel_size=128, include_eps=True)  # Custom panel size, Custom convergence criteria
    """
    def __init__(self, inputfile, panel_size=256, num_of_iter=800, conv_tol=1e-4, airfoil_name='NACA 2412'):
        """
        Initialize the solver.
        Args:
            inputfile (str): CSV file with double header and sample data.
            panel_size (int): Default number of panels for xfoil/mfoil.
            num_of_iter (int): Number of iterations for solver.
            conv_tol (float): Convergence tolerance for solver.
            airfoil_name (str): Airfoil name (e.g. 'NACA 2412').
        """
        self.dir = os.getcwd() #Gets the current working directory
        self.solver = None #Solver type (xfoil or mfoil)
        self.d = None #Number of independent variables
        self.Ns = None #Total number of samples or runs
        self.results = None #Results of the xfoil or mfoil
        self.out = None #Polar data output from xfoil or mfoil

        self.num_of_iter = int(num_of_iter) #Number of iterations for xfoil or mfoil
        self.infile = inputfile #Sample input file name
        self.xinput_file = "input_file.in" #Input file for xfoil
        self.blremove = True #Removes any .bl files that xfoil produces when running iterations
        self.polar_file = "polar_file.txt" #Polar file name for xfoil output
        self.airfoil_name = str(airfoil_name) #Airfoil name for xfoil or mfoil
        self.panel_size = int(panel_size) #Number of panels for mfoil or xfoil
        self.conv_tol = float(conv_tol) #Convergence criteria for xfoil
        
        # --- Read double header ---
        csv_path = self.dir + '/input/' + inputfile
        # Read first two rows for headers
        with open(csv_path, 'r') as f:
            first_header = f.readline().strip().split(',')
        
        self.solver = first_header[0]
        self.d = int(first_header[1])
        self.Ns = int(first_header[2])

        #Use pandas to read the CSV file
        self.df = pd.read_csv(csv_path, skiprows=1)
        self.input_names = self.df.columns.tolist()  # Get input variable names from the DataFrame
    

    def xsolve(self):
        """
        Runs xfoil for a single case using the prepared input file.
        Handles file cleanup and saves output to self.out.
        """
        polar_path = self.dir + "/output/" + self.polar_file
        input_path = self.dir + "/input/" + self.xinput_file
        

        #Deletes previous xfoil pacc files from the input directory
        if os.path.exists(polar_path):
            os.remove(polar_path)
        
        #Executes xfoil and silences the terminal prints
        with Popen(self.dir+"/src/solvers/xfoil" +" < "+self.dir+"/input/"+self.xinput_file,shell=True,stdout=DEVNULL) as process:
            try:
                process.communicate(timeout=15)
            except TimeoutExpired:
                process.kill() 
        
        #Checks if xfoil produced any data and saves it to polar_data
        if check_line_data(self.dir+"/output/"+self.polar_file,13) == True:
            polar_data = np.loadtxt(self.dir+"/output/"+self.polar_file,skiprows=12)
        else:
            polar_data = np.zeros(9)
        

        #Removes any .bl files that xfoil produces when running iterations
        if self.blremove:
            for item in os.listdir(self.dir):
                if item.endswith(".bl"):
                    os.remove(os.path.join(self.dir, item))
        
        self.out = polar_data #Saves the xfoil data to the object

        # Remove the polar file and input file after solving
        if os.path.exists(polar_path):
            os.remove(polar_path)
        if os.path.exists(input_path):
            os.remove(input_path)

    def msolve(self, msolve):
        """
        Runs mfoil for a single case.
        Sets solver parameters, runs, and saves results to self.results.
        """

        msolve.param.doplot = False #Does not print the plots
        msolve.param.rtol = self.conv_tol #Convergence tolerance for mfoil
        msolve.param.verb = 0 #Hides the prints of mfoil
        try:
            msolve.solve()
            
            if msolve.glob.conv == True:
                #Saves the coefficient of lift and moment
                self.results = np.array([msolve.post.cl,msolve.post.cm])
            else:
                #If the solver did not converge, sets the results to 0
                self.results = np.array([0,0])
        except Exception as e:
            # Handle any exception during mfoil solve
            print(f"Exception during mfoil solve: {e}")
            self.results = np.array([0,0])

    def write_xfoil_input(self, input_index, panel_size=None, include_eps=False):
        """
        Writes the xfoil input script for a single case.
        Args:
            input_index (int): Index of the sample in the DataFrame.
            panel_size (int, optional): Number of panels (default: self.panel_size).
            include_eps (bool): Whether to include eps line (for xmfoil/gci).
        Structure:
            - Loads airfoil, sets geometry, panels, flow conditions, and output file.
            - Optionally adds eps line for changing the tolerance (Not available using default xfoil).
        """
        panel_size = panel_size if panel_size is not None else self.panel_size
        input_file = open(self.dir + "/input/" + self.xinput_file, "w")
        input_file.write("PLOP\nG F\n\n")
        input_file.write(self.airfoil_name + '\n')
        input_file.write("GDES\nflap\n0.7\n999\n0.50\n")
        input_file.write(f"{self.df['flap_deflection'][input_index]}\nexec\n\n")
        input_file.write("ppar\n")
        input_file.write(f"n {panel_size}\n\n\n")
        input_file.write("OPER\nmach 0\n")
        input_file.write(f"Visc {self.df['Re'][input_index]}\n")
        input_file.write("vpar\n")
        input_file.write(f"xtr {self.df['xtr_upper'][input_index]} {self.df['xtr_lower'][input_index]}\n")
        if include_eps:
            input_file.write(f"eps {self.conv_tol}\n")
        input_file.write("\n pacc 1\n")
        input_file.write(self.dir + "/output/" + self.polar_file + "\n\n")
        input_file.write(f"ITER {self.num_of_iter}\n")
        input_file.write(f"alfa {self.df['alpha'][input_index]}\n\nquit\n")
        input_file.close()

    def multi_eval(self):
        """
        Runs the solver for all samples in the input file.
        For each sample:
            - Calls single_solve()
            - Appends results to output CSV
            - Logs failed cases to fail CSV
            - Prints progress and estimated time remaining
        """
        start = timer()
        out_path = self.dir + '/output/' + self.infile[:-4] + '_out.csv'
        fail_path = self.dir + '/output/' + self.infile[:-4] + '_fail.csv'

        # Write header if file does not exist
        if not os.path.exists(out_path):
            pd.DataFrame(columns=["Cl", "Cm"]).to_csv(out_path, index=False)

        # Loop through all samples
        for i in range(self.Ns):
            self.single_solve(i)

            if self.results[0] != 0:
                pd.DataFrame([self.results], columns=["Cl", "Cm"]).to_csv(
                    out_path, mode='a', header=False, index=False
                )
            else:
                # Write fail file only if a failed case occurs
                fail_columns = ["Index"] + self.input_names
                if not os.path.exists(fail_path):
                    pd.DataFrame(columns=fail_columns).to_csv(fail_path, index=False)
                pd.DataFrame([np.append(i, self.fx[i, :])]).to_csv(
                    fail_path, mode='a', header=False, index=False
                )
                print('Failed')

            # Print progress and estimated time remaining
            finish_time = ((timer() - start) / (i + 1)) * (self.Ns - (i + 1))
            hours = finish_time / 3600
            minutes = 60 * (hours - np.trunc(hours))
            seconds = 60 * (minutes - np.trunc(minutes))
            print('Time: {:02d}:{:02d}:{:02d} Progress: {:}/{:}'.format(
                int(np.trunc(hours)), int(np.trunc(minutes)), int(np.trunc(seconds)), i + 1, self.Ns))
            

    def run(self):
        """
        Runs all cases in the input file (calls multi_eval).
        Usage: solver_eval('input.csv').run()
        """
        self.multi_eval()
         

    def single_solve(self, input_index, panel_size=None, include_eps=False):
        """
        Runs the solver for a single sample.
        Args:
            input_index (int): Index of the sample in the DataFrame.
            panel_size (int, optional): Override panel size for this run.
            include_eps (bool): Whether to include eps line in xfoil input.
        Returns:
            None. Results are stored in self.results as [Cl, Cm].
        """
        panel = panel_size if panel_size is not None else self.panel_size

        if self.solver == 'xfoil':
            self.write_xfoil_input(input_index, panel_size=panel, include_eps=include_eps)
            self.xsolve()
            self.results = np.array([self.out[1], self.out[4]])
        elif self.solver == 'mfoil':
            mfoil_airfoil = re.sub(r'\D', '', self.airfoil_name)
            m = mfoil(naca=mfoil_airfoil, npanel=panel)
            m.geom_flap(np.float64([0.7, 0.015]), self.df['flap_deflection'][input_index])
            m.setoper(alpha=self.df['alpha'][input_index], Re=self.df['Re'][input_index], Ma=0, visc=True)
            m.vsol.xt = np.float64([self.df['xtr_upper'][input_index], self.df['xtr_lower'][input_index]])
            self.msolve(m)
        else:
            print("INCORRECT SOLVER")