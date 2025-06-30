import os
import csv
import sys
import re
import numpy as np
import pandas as pd
from subprocess import Popen, TimeoutExpired, DEVNULL
from time import monotonic as timer
from mfoil import *

#This script only focuses on taking input csv files and outputing the desired case output files
#The two cases that are supported right now are the UQ challenge problem and an mfoil and xfoil verification case


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
    """Base class for evaluating UQ problems with xfoil or mfoil."""
    def __init__(self, inputfile, panel_size=256, num_of_iter=800, conv_tol=1e-4, airfoil_name='NACA 2412'):
        self.dir = os.getcwd() #Gets the current working directory
        self.solver = None #Solver type (xfoil or mfoil)
        self.d = None #Number of independent variables
        self.Ns = None #Total number of samples or runs
        self.fx = None #Input matrix or array
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
            second_header = f.readline().strip().split(',')
        
        self.solver = first_header[0]
        self.d = int(first_header[1])
        self.Ns = int(first_header[2])
        self.input_names = second_header[:self.d]

        #Use pandas to read the CSV file
        df = pd.read_csv(csv_path, skiprows=2, names=self.input_names)
        # Remaining rows: input data
        fx = df.iloc[1:, :self.d].to_numpy(dtype=np.float64)
        # Save the input for xfoil and mfoil to the object
        if self.Ns != 1:
            self.fx = fx  # Matrix input
        else:
            self.fx = fx[0]  # Array input
        
        #Saves the sample input for xfoil and mfoil to the object
        if self.Ns != 1:
            self.fx = fx #Matrix input
        else:
            self.fx = fx[0] #Array input
    

    def xsolve(self):
        #Runs xfoil a single time
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
        

    def multi_eval(self):
       # Runs xfoil or mfoil for multiple iterations depending on the sample size
        start = timer()
        out_path = self.dir + '/output/' + self.infile[:-4] + '_out.csv'
        fail_path = self.dir + '/output/' + self.infile[:-4] + '_fail.csv'

        # Write header if file does not exist
        if not os.path.exists(out_path):
            pd.DataFrame(columns=["Cl", "Cm"]).to_csv(out_path, index=False)
        if not os.path.exists(fail_path):
            fail_columns = ["Index"] + self.input_names
            pd.DataFrame().to_csv(fail_path, index=False, header=False)

        # Loops through the total number of samples
        for i in range(self.Ns):
            self.single_solve(self.fx[i, :])

            if self.results[0] != 0:
                # Append result to output CSV
                pd.DataFrame([self.results], columns=["Cl", "Cm"]).to_csv(
                    out_path, mode='a', header=False, index=False
                )
            else:
                # Append failed case to fail CSV (index + input parameters)
                pd.DataFrame([np.append(i, self.fx[i, :])]).to_csv(
                    fail_path, mode='a', header=False, index=False
                )
                print('Failed')

            # Tracks the amount of time remaining for the multirun to finish
            finish_time = ((timer() - start) / (i + 1)) * (self.Ns - (i + 1))
            hours = finish_time / 3600
            minutes = 60 * (hours - np.trunc(hours))
            seconds = 60 * (minutes - np.trunc(minutes))
            print('Time: {:02d}:{:02d}:{:02d} Progress: {:}/{:}'.format(
                int(np.trunc(hours)), int(np.trunc(minutes)), int(np.trunc(seconds)), i + 1, self.Ns))
            

    def run(self):
        """Convenience method to run cases in one line"""
        self.multi_eval() 
         
    def single_solve(self):
        #Inheritence function that evaluates unique problems
        pass
        

class uqchallenge(solver_eval):
    """Subclass of solver_eval that focuses on the AIAA UQ challenge problem."""
    #Subclass of solver_eval that focuses on the AIAA UQ challenge problem

    #The input variables for the challenge problem were alpha, Re,
    #flap deflection, lower surface trip location, and upper surface trip location
    
    #The output variables for the challenge problem are coefficient of lift and moment
    def single_solve(self,fx):
        airfoil_name = 'NACA 2412'
        
        #If the solver uses xfoil
        if self.solver == 'xfoil':
            #Writes an input_file.in that xfoil can use to run a test case.
            
            #Turns off the plotting
            input_file = open(self.dir+"/input/"+self.xinput_file, "w")
            input_file.write("PLOP\n")
            input_file.write("G F\n\n")
            
            #Loads the airfoil
            input_file.write(self.airfoil_name + '\n')
            
            #Sets the geometric variables
            input_file.write("GDES\n")
            input_file.write("flap\n")
            input_file.write("0.7\n")
            input_file.write("999\n")
            input_file.write("0.50\n")
            input_file.write("{0}\n".format(fx[2])) #flap deflection
            input_file.write("exec\n\n")
            
            #Sets the number of panels
            input_file.write("ppar\n")
            input_file.write("n {0}\n\n\n".format(self.panel_size)) #Number of panels
            
            #Intializes the solver
            input_file.write("OPER\n")
            input_file.write("mach 0\n")
            input_file.write("Visc {0}\n".format(fx[1])) #Reynold's number
            input_file.write("vpar\n")
            input_file.write("xtr {0} {1}\n\n".format(fx[3],fx[4])) #upper and lower surface trip locations
            
            #Writes an ouput file
            input_file.write("pacc 1\n")
            input_file.write(self.dir+"/output/"+self.polar_file+"\n\n")
            
            #Solves for the case and quits xfoil
            input_file.write("ITER {0}\n".format(self.num_of_iter))
            input_file.write("alfa {0}\n".format(fx[0]))
            input_file.write("\n")
            input_file.write("quit\n")
            input_file.close()
            self.xsolve()
            
            #Saves the coefficient of lift and moment
            self.results = np.array([self.out[1],self.out[4]])
            
        #If the solver uses mfoil
        elif self.solver == "mfoil":
            #Uses mfoil to evaluate the challenge problem
            mfoil_airfoil = re.sub(r'\D', '', self.airfoil_name) #Extracts only digits from the airfoil name for mfoil

            msolve = mfoil(naca=mfoil_airfoil,npanel=self.panel_size) #Declares airfoil and number of panels
            msolve.geom_flap(np.float64([0.7,0.015]),fx[2]) #Sets the flap angle !!! not changing values
            msolve.setoper(alpha=fx[0],Re=fx[1],Ma=0,visc=True) #Sets the angle of attack, Reynold's number, and Mach number and sets the solver to viscous flows
            msolve.vsol.xt = np.float64([fx[3],fx[4]]) #Sets the upper and lower surface trip locations !!!! Not changing values
            msolve.param.doplot = False #Does not print the plots
            msolve.param.rtol = self.conv_tol #Convergence tolerance for mfoil

            
            #Hides the prints of mfoil and solves the problem
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
        else:
            print("INCORRECT SOLVER")
            
        #Returns all object variables to the main uqeval class
        return super().single_solve()



class xmfoil_verification(solver_eval):
    """Subclass for verifying and comparing xfoil and mfoil solvers."""
    def single_solve(self,fx):
        if self.solver == 'xfoil':
            airfoil_name = 'NACA 2412'

        
        conv = 1e-10
        #If the solver uses xfoil
        if self.solver == 'xfoil':
            #Writes an input_file.in that xfoil can use to run a test case.
            
            #Turns off the plotting
            input_file = open(self.dir+"/input/"+self.xinput_file, "w")
            input_file.write("PLOP\n")
            input_file.write("G F\n\n")
            
            #Loads the airfoil
            input_file.write(self.airfoil_name + '\n')
            
            #Sets the geometric variables
            input_file.write("GDES\n")
            input_file.write("flap\n")
            input_file.write("0.7\n")
            input_file.write("999\n")
            input_file.write("0.50\n")
            input_file.write("{0}\n".format(fx[2])) #flap deflection
            input_file.write("exec\n\n")
            
            #Sets the number of panels
            input_file.write("ppar\n")
            input_file.write("n {0}\n\n\n".format(self.panel_size))
            
            #Intializes the solver
            input_file.write("OPER\n")
            input_file.write("mach 0\n")
            input_file.write("Visc {0}\n".format(fx[1])) #Reynold's number
            input_file.write("vpar\n")
            input_file.write("xtr {0} {1}\n".format(fx[3],fx[4])) #upper and lower surface trip locations
            input_file.write("eps {0}\n\n".format(self.conv_tol))
            
            #Writes an ouput file
            input_file.write("pacc 1\n")
            input_file.write(self.dir+"/output/"+self.polar_file+"\n\n")
            
            #Solves for the case and quits xfoil
            input_file.write("ITER {0}\n".format(self.num_of_iter))
            input_file.write("alfa {0}\n".format(fx[0]))
            input_file.write("\n")
            input_file.write("quit\n")
            input_file.close()

            self.xsolve()
            
            #Saves the coefficient of lift and moment
            self.results = np.array([self.out[1],self.out[4]])
            
        
        elif self.solver == 'mfoil':
            #Solves the case for mfoil
            mfoil_airfoil = re.sub(r'\D', '', self.airfoil_name) #Extracts only digits from the airfoil name for mfoil

            msolve = mfoil(naca=mfoil_airfoil,npanel=self.panel_size) #number of panels
            msolve.setoper(alpha=fx[0],Re=fx[1],Ma=fx[2],visc=True) #Sets the alpha, Re, and Ma and sets teh solver to viscous
            msolve.param.doplot = False #Turns off plots
            msolve.param.verb = 0 #Hides the prints of mfoil
            msolve.param.rtol = self.conv_tol #Convergence tolerance for mfoil
            
            #solves the problem
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

        else:
            print("INCORRECT SOLVER")

        #Returns variables to solver_eval
        return super().single_solve()
