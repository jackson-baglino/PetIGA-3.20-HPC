#!/bin/bash
#SBATCH -J NASAv2-88G-2D-48h
#SBATCH -t 5-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -o "%x.o%j"
#SBATCH -e "%x.e%j"
#SBATCH --export=ALL
#SBATCH --partition=expansion

##SBATCH --mem-per-cpu=1G  # memory per CPU core
##SBATCH -p         #general partition
### /SBATCH -o slurm.%N.%j.out # STDOUT
### /SBATCH -e slurm.%N.%j.err # STDERR
echo ” ”
echo “compiling NASAv2”
echo ” ”
make NASAv2
echo ------------------------------------------------------
echo ‘Job is running on node ’; srun hostname
echo ------------------------------------------------------
echo qsub is running on $SLURM_SUBMIT_HOST
echo executing queue is $SLURM_JOB_ACCOUNT
echo working directory is $SLURM_SUBMIT_DIR
echo partition/queue is $SLURM_JOB_PARTITION
echo job identifier is $SLURM_JOB_ID
echo job name is $SLURM_JOB_NAME
echo node file is $SLURM_JOB_NODELIST
echo cluster $SLURM_CLUSTER_NAME
echo total nodes $SLURM_JOB_NUM_NODES
echo ------------------------------------------------------
echo ” ”
echo “setting up things”
echo ” ”
id=${SLURM_JOB_ID:0:9}
echo $id

name=NASAv2-88G-3D-48h_$id
folder=/central/scratch/jbaglino/$name
echo $name
echo $folder
export folder

export I_MPI_PMI_LIBRARY=/path/to/slurm/pmi/library/libpmi.so

mkdir $folder
echo "Copying files to scratch"
scp /home/jbaglino/PetIGA-3.20/demo/NASAv2.c $folder/
scp /home/jbaglino/PetIGA-3.20/demo/run_NASAv2.sh $folder/
cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR

# Define variable names to be exported -----------------------------------------
  # File names
input_dir="/home/jbaglino/PetIGA-3.20-HPC/demo/input/"
inputFile=$input_dir"grainReadFile-88_s1-10_s2-21.dat"
# inputFile=$input_dir"grainReadFile-135_s1-10_s2-21.dat"
# inputFile=$input_dir"grainReadFile-165_s1-10_s2-30.dat"
# inputFile=$input_dir"grainReadFile-10_s1-10.dat"
# inputFile=$input_dir"grainReadFile-5_s1-10.dat"
# inputFile=$input_dir"grainReadFile-2.dat"

# Define simulation parameters -------------------------------------------------
# Define dimensions
dim=2

# Converty scientic notation to decimal using bc if needed
dim=$(echo "$dim" | bc -l)

# Domain sizes
# Lx=488.4e-6                    # Domain size X -- 2 Grain
# Ly=244.2e-6                    # Domain size Y -- 2 Grain
# Lz=244.2e-6                    # Domain size Z -- 2 Grain

# Lx=0.35e-03                   # Domain size X -- 5 Grain
# Ly=0.35e-03                   # Domain size Y -- 5 Grain
# Lz=2.202e-04                  # Domain size Z -- 5 Grain

# Lx=0.5e-03                    # Domain size X -- 10 Grain
# Ly=0.5e-03                    # Domain size Y -- 10 Grain
# Lz=2.312e-04                  # Domain size Z -- 10 Grain

Lx=2.0e-3                     # Domain size X -- 88 Grain
Ly=2.0e-3                     # Domain size Y -- 88 Grain
Lz=2.509e-04                  # Domain size Z -- 88 Grain

# Lx=3.2e-3                     # Domain size X -- 165 Grain
# Ly=3.2e-3                     # Domain size Y -- 165 Grain
# Lz=0.773e-3                   # Domain size Z -- 165 Grain

# Lx=3.2e-3                     # Domain size X -- 165 Grain
# Ly=3.2e-3                     # Domain size Y -- 165 Grain
# Lz=1.0e-3                     # Domain size Z -- 165 Grain


# Number of elements
# Nx=264                        # Number of elements in X -- 2 Grain
# Ny=132                        # Number of elements in Y -- 2 Grain
# Nz=132                        # Number of elements in Z -- 2 Grain

# Nx=193                       # Number of elements in X -- 5 Grain
# Ny=193                       # Number of elements in Y -- 5 Grain
# Nz=122                        # Number of elements in Z -- 5 Grain

# Nx=270                        # Number of elements in X -- 10 Grain
# Ny=270                        # Number of elements in Y -- 10 Grain
# Nz=125                        # Number of elements in Z -- 10 Grain

# Nx=385
# Ny=385
# Nz=243

Nx=1100                       # Number of elements in X -- 88 Grain
Ny=1100                       # Number of elements in Y -- 88 Grain
Nz=138                        # Number of elements in Z -- 88 Grain

# Nx=1724                       # Number of elements in X -- 165 Grain
# Ny=1724                       # Number of elements in Y -- 165 Grain
# Nz=417                        # Number of elements in Z -- 165 Grain


# Time parameters
delt_t=1.0e-4                 # Time step
t_final=2*24*60*60              # Final time
n_out=500                     # Number of output files

# Convert scientific notation to decimal using bc
t_final=$(echo "$t_final" | bc -l)
n_out=$(echo "$n_out" | bc -l)

# Other parameters
humidity=0.98                 # Relative humidity
temp=-30.0                    # Temperature

# Initial temperature gradients
grad_temp0X=0.0               # Initial temperature gradient X
grad_temp0Y=3.0               # Initial temperature gradient Y
grad_temp0Z=0.0               # Initial temperature gradient Z

# Convert scientific notation gradients to decimal using bc if needed
grad_temp0X=$(echo "$grad_temp0X" | bc -l)
grad_temp0Y=$(echo "$grad_temp0Y" | bc -l)
grad_temp0Z=$(echo "$grad_temp0Z" | bc -l)

# Export variables
export folder input_dir inputFile Lx Ly Lz Nx Ny Nz delt_t t_final n_out \
    humidity temp grad_temp0X grad_temp0Y grad_temp0Z dim


echo " "
echo "running NASAv2"
echo " "

mpiexec ./NASAv2 -initial_cond -initial_PFgeom -snes_rtol 1e-3 -snes_stol 1e-6 \
-snes_max_it 6 -ksp_gmres_restart 150 -ksp_max_it 500 -ksp_converged_maxits 1 \
-ksp_converged_reason -snes_converged_reason -snes_linesearch_monitor  \
-snes_linesearch_type basic | tee $folder/outp.txt

# Create descriptive file ------------------------------------------------------
echo "----- SIMULATION PARAMETERS -----" > $folder/sim_params.dat
echo "Input file: $inputFile" >> $folder/sim_params.dat

echo " " >> $folder/sim_params.dat

echo "Dimensions:" >> $folder/sim_params.dat
echo "dim = $dim" >> $folder/sim_params.dat

echo " " >> $folder/sim_params.dat

echo "Domain sizes:" >> $folder/sim_params.dat
echo "Lx = $Lx" >> $folder/sim_params.dat
echo "Ly = $Ly" >> $folder/sim_params.dat
echo "Lz = $Lz" >> $folder/sim_params.dat

echo " " >> $folder/sim_params.dat


echo "Number of elements:" >> $folder/sim_params.dat
echo "Nx = $Nx" >> $folder/sim_params.dat
echo "Ny = $Ny" >> $folder/sim_params.dat
echo "Nz = $Nz" >> $folder/sim_params.dat

echo " " >> $folder/sim_params.dat

echo "Time parameters:" >> $folder/sim_params.dat
echo "delt_t = $delt_t" >> $folder/sim_params.dat
echo "t_final = $t_final" >> $folder/sim_params.dat

echo " " >> $folder/sim_params.dat

echo "State parameters:" >> $folder/sim_params.dat
echo "humidity = $humidity" >> $folder/sim_params.dat
echo "temp = $temp" >> $folder/sim_params.dat

echo " " >> $folder/sim_params.dat

echo "Initial temperature gradients:" >> $folder/sim_params.dat
echo "grad_temp0X = $grad_temp0X" >> $folder/sim_params.dat
echo "grad_temp0Y = $grad_temp0Y" >> $folder/sim_params.dat
echo "grad_temp0Z = $grad_temp0Z" >> $folder/sim_params.dat

echo "-------------------------------------------------------------------------"

echo "done"
