#!/bin/bash
#SBATCH -J permafrost-run
#SBATCH -A rubyfu
#SBATCH -t 7-00:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH -o "output_files/%x.o%j"
#SBATCH -e "output_files/%x.e%j"
#SBATCH --export=ALL
#SBATCH --partition=expansion
#SBATCH --mail-user=jbaglino@caltech.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# ==============================
# Configuration
# ==============================
BASE_DIR="/home/jbaglino/PetIGA-3.20-HPC/projects/permafrost"
INPUT_DIR="$BASE_DIR/inputs"
OUTPUT_BASE="/resnick/scratch/jbaglino/permafrost"
EXEC_FILE="$BASE_DIR/permafrost"
inputFile="$INPUT_DIR/circle_data.csv"

# Output Folder
timestamp=$(date +%Y-%m-%d__%H.%M.%S)
folder="$OUTPUT_BASE/permafrost_$timestamp"
mkdir -p "$folder"

# Parameters
Lx=1.0e-03
Ly=1.0e-03
Lz=1.0e-03
Nx=550
Ny=550
Nz=550
eps=9.09629658751972e-07
delt_t=1.0e-4
t_final=$((52*7*24*60*60))
n_out=100
humidity=0.75
temp=-20.0
grad_temp0X=0.0
grad_temp0Y=0.003
grad_temp0Z=0.0
dim=2

export Lx Ly Lz Nx Ny Nz eps delt_t t_final n_out \
       humidity temp grad_temp0X grad_temp0Y grad_temp0Z dim inputFile folder

# ==============================
# Logging Info
# ==============================
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Output folder: $folder"
echo "[INFO] Input file: $inputFile"
echo "[INFO] Running on nodes: $SLURM_JOB_NODELIST"
echo "[INFO] Starting simulation..."

# ==============================
# Compilation (optional)
# ==============================
cd "$BASE_DIR" || exit 1
make permafrost || { echo "❌ Compilation failed"; exit 2; }

# ==============================
# Execution
# ==============================
mpiexec "$EXEC_FILE" -initial_PFgeom -temp_initial \
    -snes_rtol 1e-3 -snes_stol 1e-6 -snes_max_it 7 \
    -ksp_gmres_restart 150 -ksp_max_it 1000 \
    -ksp_converged_reason -snes_converged_reason \
    -snes_linesearch_monitor -snes_linesearch_type basic \
    | tee "$folder/outp.txt"

# ==============================
# Archiving and Logs
# ==============================
cp "$inputFile" "$folder/"
cp "$BASE_DIR/src/permafrost.c" "$folder/"
cp "$BASE_DIR/scripts/run_permafrost.sh" "$folder/"

# Save simulation parameters
cat << EOF > "$folder/sim_params.dat"
----- SIMULATION PARAMETERS -----
Input file: $inputFile

Dimensions:
dim = $dim

Interface width:
eps = $eps

Domain sizes:
Lx = $Lx
Ly = $Ly
Lz = $Lz

Number of elements:
Nx = $Nx
Ny = $Ny
Nz = $Nz

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

# Completion Message
echo "[INFO] Simulation completed. Results saved in $folder"
