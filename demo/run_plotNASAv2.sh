#!/bin/bash
#SBATCH -J plotResults_JOB_NAME
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH -o "output_files/plotting/%x.o%j"
#SBATCH -e "output_files/plotting/%x.e%j"
#SBATCH --export=ALL
#SBATCH --partition=expansion
#SBATCH --mem-per-cpu=1G

# Define the job name
JOB_NAME=""

# Compilation
echo ""
echo "------------------------------------------------------"
echo "Job is running on node"; srun hostname
echo "------------------------------------------------------"
echo "qsub is running on $SLURM_SUBMIT_HOST"
echo "executing queue is $SLURM_JOB_ACCOUNT"
echo "working directory is $SLURM_SUBMIT_DIR"
echo "partition/queue is $SLURM_JOB_PARTITION"
echo "job identifier is $SLURM_JOB_ID"
echo "job name is $SLURM_JOB_NAME"
echo "node file is $SLURM_JOB_NODELIST"
echo "cluster $SLURM_CLUSTER_NAME"
echo "total nodes $SLURM_JOB_NUM_NODES"
echo "------------------------------------------------------"
echo ""
echo "setting up things"
echo ""


inputFolder=""
outputFolder=""

echo "inputFolder: $inputFolder"
echo "outputFolder: $outputFolder"

export inputFolder outputFolder

# Plot results to vtk and vtp files


# Plot SSA Evolution




















# #!/bin/zsh

# if [[ -n $1 ]]; then
#     echo "Copying run_plotNASAv2.py to: $1"
#     dir=/central/scratch/jbaglino/$1
#     cp plotNASA.py $dir
#     cp writeNASA2CSV.py $dir

#     exec_dir=/home/jbaglino/PetIGA-3.20-HPC/demo

#     echo " "
#     echo $exec_dir
#     echo " "

#     cp $exec_dir"/plotSSA.py" $dir
#     echo " "
#     echo "Copying plotSSA.py to: $1"
#     echo " "
#     echo "Working directory: $(pwd)"
#     echo " "
#     echo "Executing python script"
#     echo " "

#     mkdir $dir/vtkOut

#     python3.11 ~/SimulationResults/DrySed_Metamorphism/NASAv2/$1/plotNASA.py
#     # python3.11 ~/SimulationResults/DrySed_Metamorphism/NASAv2/$1/writeNASA2CSV.py

#     echo "Plotting SSA and Porosity"
#     echo " "
#     echo "Calling from: $1"

#     python3.11 ~/SimulationResults/DrySed_Metamorphism/NASAv2/$1/plotSSA.py
#     # python3.11 ~/SimulationResults/DrySed_Metamorphism/NASAv2/$1/plotPorosity.py
# else
#     echo "No inputs are given. Assume we are already in the results folder"
#     echo " "

#     echo "Creating vtkOut and stlOut directories"
#     echo " "
    
#     mkdir -p vtkOut
#     mkdir -p stlOut
#     cp -r -p stlOut stl_Out-copy

#     python3.11 plotNASA.py
    
#     python3.11 plotSSA.py
#     python3.11 plotPorosity.py

#     python3.11 rename_STL_files.py
# fi

