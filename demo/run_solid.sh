#!/bin/zsh
 
echo " "
echo "compiling"
echo " "
make Solid3ph_flow
 
echo " "
echo "running Solid3ph_flow"
echo " "

cp Solid3ph_flow.c ~/Simulation_results/solidif_results/

~/petsc-3.20/lib/petsc/bin/petscmpiexec -np 4 ./Solid3ph_flow -initial_condition  -initial_PFgeom -snes_rtol 1e-3 -snes_stol 1e-6 -snes_max_it 4 -ksp_gmres_restart 150 -ksp_max_it 500 -ksp_converged_reason -ksp_converged_maxits 1 -snes_converged_reason  -snes_linesearch_monitor -snes_linesearch_type basic | tee ~/Simulation_results/solidif_results/outp.txt
