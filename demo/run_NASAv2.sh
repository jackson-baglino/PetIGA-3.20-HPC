#!/bin/bash
#SBATCH -J NASAv2-500Go-3D-2D-T85K-hum70
#SBATCH -A rubyfu
#SBATCH -t 5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=25
#SBATCH --cpus-per-task=1
#SBATCH -o "output_files/%x.o%j"
#SBATCH -e "output_files/%x.e%j"
#SBATCH --export=ALL
#SBATCH --partition=expansion
#SBATCH --mem-per-cpu=2G

##############################################
# CONFIGURATION
##############################################

# General settings
BASE_DIR="/home/jbaglino/PetIGA-3.20-HPC/demo"
input_dir="$BASE_DIR/input"
output_dir="/central/scratch/jbaglino"
exec_file="./NASAv2"

# Job-specific settings
humidity=0.70
temp=-188.0

# Define input file (uncomment the desired file)
# inputFile="$input_dir/grainReadFile-2.dat"
# inputFile="$input_dir/grainReadFile-2_Molaro.dat"
# inputFile="$input_dir/grainReadFile-5_s1-10.dat"
# inputFile="$input_dir/grainReadFile-10_s1-10.dat"
inputFile="$input_dir/grainReadFile_3D-500_s1-10.dat"

##############################################
# FUNCTIONS
##############################################

# Validate critical inputs
validate_inputs() {
    if [[ -z "$inputFile" ]]; then
        echo "[ERROR] No input file specified. Exiting."
        exit 1
    fi

    if [[ ! -f "$inputFile" ]]; then
        echo "[ERROR] Input file '$inputFile' does not exist. Exiting."
        exit 1
    fi
}

# Set parameters based on input file
set_parameters() {
    case "$inputFile" in
        *grainReadFile-2.dat)
            Lx=488.4e-6
            Ly=244.2e-6
            Lz=244.2e-6

            Nx=269
            Ny=135
            Nz=135

            eps=9.096e-07
            ;;
        *grainReadFile-2_Molaro.dat)
            Lx=0.0002424
            Ly=0.0003884
            Lz=0.0002424

            Nx=134
            Ny=214
            Nz=134

            eps=9.096e-07
            ;;
        *grainReadFile-5_s1-10.dat)
            Lx=0.35e-03; Ly=0.35e-03; Lz=2.202e-04
            Nx=193; Ny=193; Nz=122
            eps=9.096e-07
            ;;

        *grainReadFile_3D-500_s1-10.dat)
            Lx=1.5e-03; Ly=1.5e-03; Lz=1.5e-03
            Nx=478; Ny=478; Nz=478
            eps=1.56979924263831e-06;
            ;;
        *)
            echo "[WARNING] No matching parameters for '$inputFile'. Using defaults."
            Lx=0.0002424; Ly=0.0003884; Lz=0.0002424
            Nx=200; Ny=200; Nz=720
            eps=9.096e-07;
            ;;
    esac

    # Shared parameters
    dim=3
    grad_temp0X=0.0
    grad_temp0Y=0.001
    grad_temp0Z=0.0

    t_final=2.0*24.0*60.0*60.0
    delt_t=1.0e-4
    n_out=0


    # Export all the variables
    export Lx Ly Lz Nx Ny Nz eps delt_t t_final n_out dim grad_temp0X grad_temp0Y grad_temp0Z \
           humidity temp inputFile folder
}

# Log system and job information
log_job_info() {
    echo "------------------------------------------------------"
    echo "[INFO] Job Information:"
    echo "Running on node:"; srun hostname
    echo "Submit host: $SLURM_SUBMIT_HOST"
    echo "Account: $SLURM_JOB_ACCOUNT"
    echo "Working directory: $SLURM_SUBMIT_DIR"
    echo "Partition/queue: $SLURM_JOB_PARTITION"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Job name: $SLURM_JOB_NAME"
    echo "Node list: $SLURM_JOB_NODELIST"
    echo "Cluster: $SLURM_CLUSTER_NAME"
    echo "Total nodes: $SLURM_JOB_NUM_NODES"
    echo "------------------------------------------------------"
}

# Create unique output folder
setup_output_folder() {
    id=${SLURM_JOB_ID:0:9}
    name="${SLURM_JOB_NAME}_${id}"
    folder="$output_dir/$name"
    mkdir -p "$folder"
    export folder
    echo "[INFO] Output folder created: $folder"
}

# Compile NASAv2
compile_program() {
    echo "[INFO] Compiling NASAv2..."
    make NASAv2 || { echo "[ERROR] Compilation failed! Exiting."; exit 2; }
}

# Run the program
run_program() {
    echo "[INFO] Running NASAv2..."
    export I_MPI_PMI_LIBRARY=/path/to/slurm/pmi/library/libpmi.so
    mpiexec -- "$exec_file" -initial_cond -initial_PFgeom -snes_rtol 1e-3 -snes_stol 1e-6 \
     -snes_max_it 6 -ksp_gmres_restart 150 -ksp_max_it 500 -ksp_converged_maxits 1 \
     -ksp_converged_reason -snes_converged_reason -snes_linesearch_monitor \
     -snes_linesearch_type basic | tee "$folder/outp.txt"
}

# Save parameters to a file
save_simulation_parameters() {
    cat << EOF > "$folder/sim_params.dat"
----- SIMULATION PARAMETERS -----
Input file: $inputFile

Dimensions:
dim = $dim

Domain sizes:
Lx = $Lx
Ly = $Ly
Lz = $Lz

Number of elements:
Nx = $Nx
Ny = $Ny
Nz = $Nz

Interface width:
eps = $eps

Time parameters:
delt_t = $delt_t
t_final = $t_final

State parameters:
humidity = $humidity
temp = $temp

Initial temperature gradients:
grad_temp0X = $grad_temp0X
grad_temp0Y = $grad_temp0Y
grad_temp0Z = $grad_temp0Z

EOF
    echo "[INFO] Simulation parameters saved to: $folder/sim_params.dat"
}

# Cleanup temporary files
cleanup() {
    echo "[INFO] Cleaning up temporary files..."
    rm -rf "$folder/tmp/*"
}

##############################################
# MAIN SCRIPT
##############################################

# Validate inputs
validate_inputs

# Setup output folder
setup_output_folder

# Set parameters
set_parameters

# Log job info
log_job_info

# Copy necessary files to the output folder
echo "[INFO] Copying files to output folder..."
scp "$inputFile" "$folder/"
scp "$BASE_DIR/NASAv2.c" "$folder/"
scp "$BASE_DIR/run_NASAv2.sh" "$folder/"

# Compile program
compile_program

# Run program
run_program

# Save parameters to file
save_simulation_parameters

# Cleanup
cleanup

echo "[INFO] Simulation completed successfully. Results saved in $folder."