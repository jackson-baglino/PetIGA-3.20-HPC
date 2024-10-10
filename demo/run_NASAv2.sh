#!/bin/bash
#SBATCH -J NASAv2-2G-3D-48h-T20-hum988
#SBATCH -t 5-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1
#SBATCH -o "output_files/%x.o%j"
#SBATCH -e "output_files/%x.e%j"
#SBATCH --export=ALL
#SBATCH --partition=expansion
#SBATCH --mem-per-cpu=1G

# Define the job name
JOB_NAME="NASAv2-2G-3D-48h-T20-hum98"

# Other parameters
humidity=0.98                 # Relative humidity
temp=-20.0                    # Temperature


# Compilation
echo ""
echo "compiling NASAv2"
echo ""
make NASAv2
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

# Create a unique folder name using the job name and job ID
id=${SLURM_JOB_ID:0:9}
name="${JOB_NAME}_${id}"
folder="/central/scratch/jbaglino/$name"
echo $name
echo $folder
export folder

# The rest of your script goes here, utilizing the $folder variable
export I_MPI_PMI_LIBRARY=/path/to/slurm/pmi/library/libpmi.so

mkdir $folder
echo "Copying files to scratch"
scp /home/jbaglino/PetIGA-3.20-HPC/demo/NASAv2.c $folder/
scp /home/jbaglino/PetIGA-3.20-HPC/demo/run_NASAv2.sh $folder/
cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR

# Define variable names to be exported -----------------------------------------
  # File names
input_dir="/home/jbaglino/PetIGA-3.20-HPC/demo/input/"
# inputFile=$input_dir"grainReadFile-2.dat"
inputFile=$input_dir"grainReadFile-2_Molaro.dat"
# inputFile=$input_dir"grainReadFile-5_s1-10.dat"
# inputFile=$input_dir"grainReadFile-10_s1-10.dat"
# inputFile=$input_dir"grainReadFile_3D-42_s1-10.dat"
# inputFile=$input_dir"grainReadFile-45_MOLARO_s2-10.dat"
# inputFile=$input_dir"grainReadFile-27_MOLARO_s2-10.dat"
# inputFile=$input_dir"grainReadFile-88_s1-10_s2-21.dat"
# inputFile=$input_dir"grainReadFile-135_s1-10_s2-30.dat"
# inputFile=$input_dir"grainReadFile-165_s1-10_s2-30.dat"


# Define simulation parameters -------------------------------------------------
# Define dimensions
dim=3


# Converty scientic notation to decimal using bc if needed
dim=$(echo "$dim" | bc -l)

# Domain sizes
Lx=488.4e-6                   # Domain size X -- 2 Grain
Ly=244.2e-6                   # Domain size Y -- 2 Grain
Lz=244.2e-6                   # Domain size Z -- 2 Grain

# Lx=3.0300e-04                   # Domain size X -- 2 Grain (Molaro)
# Ly=3.8280e-04                   # Domain size Y -- 2 Grain (Molaro)
# Lz=3.0300e-04                   # Domain size Z -- 2 Grain (Molaro)

# Lx=0.35e-03                   # Domain size X -- 5 Grain
# Ly=0.35e-03                   # Domain size Y -- 5 Grain
# Lz=2.202e-04                  # Domain size Z -- 5 Grain

# Lx=0.5e-03                    # Domain size X -- 10 Grain
# Ly=0.5e-03                    # Domain size Y -- 10 Grain
# Lz=2.422e-04                  # Domain size Z -- 10 Grain

# Lx=0.75e-03                   # Domain size X -- 27 Grain
# Ly=0.75e-03                   # Domain size Y -- 27 Grain
# Lz=0.000242175903182621       # Domain size Z -- 27 Grain

# Lx=0.5e-03                    # Domain size X -- 42 Grain (3D)
# Ly=0.5e-03                    # Domain size Y -- 42 Grain (3D)
# Lz=0.5e-03                    # Domain size Z -- 42 Grain (3D)

# Lx=1.5e-3                     # Domain size X -- 45 Grain
# Lx=1.5e-3                     # Domain size X -- 45 Grain
# Lx=1.5e-3                     # Domain size X -- 45 Grain

# Lx=2.0e-3                     # Domain size X -- 88 Grain
# Ly=2.0e-3                     # Domain size Y -- 88 Grain
# Lz=0.6021e-3                  # Domain size Z -- 88 Grain

# Lx=3.2e-3                     # Domain size X -- 135/165 Grain
# Ly=3.2e-3                     # Domain size Y -- 135/165 Grain
# Lz=1.0e-3                     # Domain size Z -- 135/165 Grain


# Number of elements
Nx=264                        # Number of elements in X -- 2 Grain
Ny=132                        # Number of elements in Y -- 2 Grain
Nz=132                        # Number of elements in Z -- 2 Grain

# Nx=537                          # Number of elements in X -- 2 Grain (High-res)
# Ny=269                          # Number of elements in Y -- 2 Grain (High-res)
# Nz=269                          # Number of elements in Z -- 2 Grain (High-res)

# Nx=167                        # Domain size X -- 2 Grain (Molaro)
# Ny=211                        # Domain size Y -- 2 Grain (Molaro)
# Nz=167                        # Domain size Z -- 2 Grain (Molaro)

# Nx=275                        # Number of elements in X -- 42 Grain (3D)
# Ny=275                        # Number of elements in Y -- 42 Grain (3D)
# Nz=275                        # Number of elements in Z -- 42 Grain (3D)

# Nx=193                        # Number of elements in X -- 5 Grain
# Ny=193                        # Number of elements in Y -- 5 Grain
# Nz=122                        # Number of elements in Z -- 5 Grain

# Nx=270                        # Number of elements in X -- 10 Grain
# Ny=270                        # Number of elements in Y -- 10 Grain
# Nz=131                        # Number of elements in Z -- 10 Grain

# Nx=413                        # Number of elements in X -- 27 Grain
# Ny=413                        # Number of elements in Y -- 27 Grain
# Nz=134                        # Number of elements in Z -- 27 Grain

# Nx=825                        # Number of elements in X -- 45 Grain
# Ny=825                        # Number of elements in Y -- 45 Grain
# Nz=325                        # Number of elements in Z -- 45 Grain

# Nx=1078                       # Number of elements in X -- 88 Grain
# Ny=1078                       # Number of elements in Y -- 88 Grain
# Nz=325                        # Number of elements in Z -- 88 Grain

# Nx=1724                       # Number of elements in X -- 135/165 Grain
# Ny=1724                       # Number of elements in Y -- 135/165 Grain
# Nz=549                        # Number of elements in Z -- 135/165 Grain

# Nx=200
# Ny=200
# Nz=720

# Interface width
eps=9.1e-07                       # Interface width (2 Grain)
# eps=9.28146307269926e-07			  # Interface width
# eps=9.09629658751972e-07        # Interface width (27 Grain)


# Time parameters
delt_t=1.0e-4                     # Time step
t_final=2*24*60*60                # Final time
n_out=200                         # Number of output files
# t_final=2*24*60*60                # Final time
# n_out=1100                        # Number of output files
# t_final=1.0e-4                    # Final time (TEST)
# n_out=1                           # Number of output files (TEST)


# Convert scientific notation to decimal using bc
t_final=$(echo "$t_final" | bc -l)
n_out=$(echo "$n_out" | bc -l)


# Initial temperature gradients
grad_temp0X=0.0               # Initial temperature gradient X
grad_temp0Y=0.03               # Initial temperature gradient Y
grad_temp0Z=0.0               # Initial temperature gradient Z


# Convert scientific notation gradients to decimal using bc if needed
grad_temp0X=$(echo "$grad_temp0X" | bc -l)
grad_temp0Y=$(echo "$grad_temp0Y" | bc -l)
grad_temp0Z=$(echo "$grad_temp0Z" | bc -l)


# Export variables
export folder input_dir inputFile Lx Ly Lz Nx Ny Nz delt_t t_final n_out \
    humidity temp grad_temp0X grad_temp0Y grad_temp0Z dim eps


echo " "
echo "running NASAv2"
echo " "


# Run NASAv2 -------------------------------------------------------------------
mpiexec -- ./NASAv2 -initial_cond -initial_PFgeom -snes_rtol 1e-3 -snes_stol 1e-6 \
-snes_max_it 6 -ksp_gmres_restart 150 -ksp_max_it 500 -ksp_converged_maxits 1 \
-ksp_converged_reason -snes_converged_reason -snes_linesearch_monitor \
-snes_linesearch_type basic | tee $folder/outp.txt


# Create descriptive file ------------------------------------------------------
echo "----- SIMULATION PARAMETERS -----" > $folder/sim_params.dat
echo "Input file: $inputFile" >> $folder/sim_params.dat
echo " " >> $folder/sim_params.dat

echo "Dimensions:" >> $folder/sim_params.dat
echo "dim = $dim" >> $folder/sim_params.dat
echo " " >> $folder/sim_params.dat

echo "Interface wiedth:" >> $folder/sim_params.dat
echo "eps = $eps" >> $folder/sim_params.dat
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
