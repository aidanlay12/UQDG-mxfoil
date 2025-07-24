"""
Tutorial: Grid Convergence Index (GCI) Evaluation with XFOIL/mfoil

This script demonstrates how to:
- Generate mesh refinement samples for GCI analysis.
- Run the solver (XFOIL or mfoil) on each mesh.
- Perform GCI analysis to estimate numerical uncertainty and error.

Steps:
1. Define a range of panel sizes for mesh refinement.
2. Generate GCI sample input file at a fixed evaluation point.
3. Run the solver for each mesh.
4. Perform GCI analysis for a specified starting panel size and report error/uncertainty for a target panel size.

Modify the 'solver' argument in smp.sample(...) to 'xfoil' or 'mfoil' as needed.
"""

import UQDGmxfoil.sample as smp
import UQDGmxfoil.solver_eval as xmeval
import UQDGmxfoil.uq_analysis as uq
import numpy as np

# Step 1: Define panel sizes for mesh refinement
num_of_panels = np.arange(32, 1028, 2)
num_of_meshes = np.size(num_of_panels)

# Step 2: Generate GCI sample input file for a fixed evaluation point
# Change solver='xfoil' to solver='mfoil' to use mfoil instead
smp.sample(num_samples=num_of_meshes, 
           solver='xfoil').create_gci_samples(csv_name='gci_samples.csv', 
                                              evaluation_point=[0, 500000, 0, 0.3, 0.7], 
                                              panel_sizes=num_of_panels)

# Step 3: Run the solver for each mesh
xmeval.solver_eval('gci_samples.csv').run()

# Step 4: Perform GCI analysis and print results
uq.uq_analysis().gci_analysis(csv_file='gci_samples.csv', 
                              starting_panel_size=210, 
                              panel_size_used=256)