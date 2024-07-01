#!/bin/zsh

echo " "
echo "compiling"
echo " "
make NASA_read

# add name folder accordingly:    <---------------------------------------------
name=res_$(date +%Y-%m-%d__%H.%M.%S)
dir=/Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/NASA_read
folder=$dir/$name

mkdir $folder/

export folder

cp NASA_read.c  $folder
cp run_NASA_read.sh $folder

echo " "
echo "Calling ./NASA_read"
echo " "

mpiexec -np 4 ./NASA_read  -initial_PFgeom \
-temp_initial -snes_rtol 1e-3 -snes_stol 1e-6 -snes_max_it 7 \
-ksp_gmres_restart 150 -ksp_max_it 1000  -ksp_converged_reason \
-snes_converged_reason  -snes_linesearch_monitor \
-snes_linesearch_type basic | tee /Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/NASA_read/outp.txt


echo " "
echo "making directory" $folder
echo " "

mv $dir*.dat $folder
mv $dir/outp.txt $folder

echo "Queing plotNASA.py"
./run_plotNASA_read.sh $name

echo "-------------------------------------------------------------------------"
echo "Done!"
echo " "

