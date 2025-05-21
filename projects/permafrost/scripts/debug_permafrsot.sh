#!/bin/bash
#SBATCH -J permafrost-debug
#SBATCH -A rubyfu
#SBATCH -t 52-00:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH -o "output_files/%x.o%j"
#SBATCH -e "output_files/%x.e%j"
#SBATCH --export=ALL
#SBATCH --partition=expansion
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=jbaglino@caltech.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR="/home/jbaglino/PetIGA-3.20-HPC/projects/permafrost/"
input_dir="${BASE_DIR}inputs"
inputFile="$input_dir/circle_data.csv"

# -----------------------------
# FUNCTIONS
# -----------------------------
validate_inputs() {
    if [[ ! -f "$inputFile" ]]; then
        echo "[ERROR] Input file '$inputFile' not found. Exiting."
        exit 1
    fi
}

set_parameters() {
    Lx=0.5e-03; Ly=0.5e-03; Lz=0.5e-03
    Nx=275; Ny=275; Nz=275
    eps=9.09629658751972e-07

    delt_t=1.0e-4
    t_final=4.0e-4
    n_out=10
    t_final=$(printf "%.10f" "$t_final")

    humidity=0.70
    temp=-20.0
    grad_temp0X=0.0
    grad_temp0Y=3.0
    grad_temp0Z=0.0
    dim=2

    timestamp=$(date +%Y-%m-%d__%H.%M.%S)
    folder="/resnick/scratch/jbaglino/permafrost_debug_$timestamp"
    mkdir -p "$folder"

    export folder inputFile Lx Ly Lz Nx Ny Nz delt_t t_final n_out \
           humidity temp grad_temp0X grad_temp0Y grad_temp0Z dim eps
}

log_info() {
    echo "[INFO] Running debug job on $(hostname)"
    echo "[INFO] Output folder: $folder"
}

compile_debug() {
    echo "[INFO] Compiling permafrost with debug flags..."
    make clean
    if ! make BUILD=debug permafrost; then
        echo "[ERROR] Compilation failed!"
        exit 2
    fi
}

run_debug() {
    echo "[INFO] Launching MPI debug session..."
    srun gdb --args "$exec_file" -initial_PFgeom -temp_initial
}

save_metadata() {
    cp "$inputFile" "$folder/"
    cp "$BASE_DIR/src/permafrost.c" "$folder/"
    cp "$BASE_DIR/scripts/run_permafrost.sh" "$folder/"
    cat << EOF > "$folder/sim_params.dat"
Input file: $inputFile
Dimensions: $dim
Domain: $Lx × $Ly × $Lz
Grid: $Nx × $Ny × $Nz
Time: dt=$delt_t, t_final=$t_final
Gradients: dT/dx=$grad_temp0X, dT/dy=$grad_temp0Y, dT/dz=$grad_temp0Z
EOF
}

# -----------------------------
# MAIN SCRIPT
# -----------------------------
validate_inputs
set_parameters
log_info
compile_debug
save_metadata
run_debug

echo "[INFO] Debug job complete. Files in $folder"