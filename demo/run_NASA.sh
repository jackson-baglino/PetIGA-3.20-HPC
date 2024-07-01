#!/bin/zsh

echo " "
echo "compiling"
echo " "
make NASA

# add name folder accordingly:    <---------------------------------------------
name=res_$(date +%Y-%m-%d__%H.%M.%S)
dir=/Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/NASA/
folder=$dir/$name

mkdir $folder/

export folder

cp NASA.c  $folder
cp run_NASA.sh $folder

echo " "
echo "Calling ./NASA"
echo " "

mpiexec -np 6 ./NASA  -initial_PFgeom \
-temp_initial -snes_rtol 1e-3 -snes_stol 1e-6 -snes_max_it 7 \
-ksp_gmres_restart 150 -ksp_max_it 1000  -ksp_converged_reason \
-snes_converged_reason  -snes_linesearch_monitor \
-snes_linesearch_type basic | tee /Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/NASA/outp.txt


echo " "
echo "making directory" $folder
echo " "

# mv $dir/*.dat $folder
mv $dir/outp.txt $folder
cp run_plotNASA_panda.sh $folder

echo "Queing plotNASA.py"
./run_plotNASA_panda.sh $name

echo "-------------------------------------------------------------------------"
echo "Done!"
echo " "

