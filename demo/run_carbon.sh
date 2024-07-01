#!/bin/zsh
 
echo " "
echo "compiling"
echo " "
make Carbon
 
echo " "
echo "running carbon"
echo " "

cp Carbon.c ~/Simulation_results/carbon_results/

~/petsc-3.20/lib/petsc/bin/petscmpiexec -np 4 ./Carbon -iga_view  -snes_rtol 5e-4 -snes_stol 1e-6 -snes_max_it 5 -ksp_gmres_restart 150 -ksp_max_it 500 -ksp_converged_reason -snes_converged_reason  -snes_linesearch_monitor -snes_linesearch_type basic | tee ~/Simulation_results/carbon_results/outp.txt
