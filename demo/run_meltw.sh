#!/bin/zsh
 
echo " "
echo "compiling"
echo " "
make Meltwater
 
echo " "
echo "running meltwater"
echo " "

cp Meltwater.c ~/Simulation_results/meltw_results/

~/petsc-3.20/lib/petsc/bin/petscmpiexec -np 4 ./Meltwater -iga_view -melt_initial  -snes_rtol 5e-4 -snes_stol 1e-6 -snes_max_it 5 -ksp_gmres_restart 150 -ksp_max_it 500 -ksp_converged_reason -snes_converged_reason  -snes_linesearch_monitor -snes_linesearch_type basic | tee ~/Simulation_results/meltw_results/outp.txt
