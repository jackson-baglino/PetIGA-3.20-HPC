#!/bin/zsh

set -euo pipefail
trap 'echo "❌ Error on line $LINENO"; exit 1' ERR

################################################################################
# permafrost Dry Snow Metamorphism Simulation Script
# This script compiles and runs the permafrost model with user-defined inputs,
# creates output directories, saves metadata, and post-processes results.
################################################################################

################################################################################
# Create output folder based on timestamp and title
################################################################################
create_folder() {
    name="$title$(date +%Y-%m-%d__%H.%M.%S)"
    dir="/Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/permafrost"
    folder="$dir/$name"

    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi

    mkdir -p "$folder"
}

################################################################################
# Compile simulation code
################################################################################
compile_code() {
    echo "Compiling..."
    make all
}

################################################################################
# Export simulation parameters to CSV file
################################################################################
write_parameters_to_csv() {
    csv_file="$folder/simulation_parameters.csv"
    echo "Variable,Value" > "$csv_file"
    echo "folder,$folder" >> "$csv_file"
    echo "inputFile,$inputFile" >> "$csv_file"
    echo "title,$title" >> "$csv_file"
    echo "Lx,$Lx" >> "$csv_file"
    echo "Ly,$Ly" >> "$csv_file"
    echo "Lz,$Lz" >> "$csv_file"
    echo "Nx,$Nx" >> "$csv_file"
    echo "Ny,$Ny" >> "$csv_file"
    echo "Nz,$Nz" >> "$csv_file"
    echo "delt_t,$delt_t" >> "$csv_file"
    echo "t_final,$t_final" >> "$csv_file"
    echo "n_out,$n_out" >> "$csv_file"
    echo "humidity,$humidity" >> "$csv_file"
    echo "temp,$temp" >> "$csv_file"
    echo "grad_temp0X,$grad_temp0X" >> "$csv_file"
    echo "grad_temp0Y,$grad_temp0Y" >> "$csv_file"
    echo "grad_temp0Z,$grad_temp0Z" >> "$csv_file"
    echo "dim,$dim" >> "$csv_file"
    echo "eps,$eps" >> "$csv_file"
}

################################################################################
# Load input file and set grid size, domain, and epsilon based on selected input
################################################################################
set_parameters() {
    input_dir="/Users/jacksonbaglino/PetIGA-3.20/projects/dry_snow_metamorphism/inputs/"
    inputFile="${input_dir}${filename}"

    cp "$inputFile" "$folder"
    echo "Selected input file: $inputFile"

    Lx=0.5e-03
    Ly=0.5e-03
    Lz=0.5e-03
    Nx=275
    Ny=275
    Nz=275
    eps=9.09629658751972e-07

    export folder inputFile title Lx Ly Lz Nx Ny Nz delt_t t_final n_out \
        humidity temp grad_temp0X grad_temp0Y grad_temp0Z dim eps
}

################################################################################
# Run the simulation using MPI
################################################################################
run_simulation() {
    echo "Running simulation..."
    echo "Nx: $Nx"
    echo "Ny: $Ny"
    echo "Nz: $Nz"
    echo "Lx: $Lx"
    echo "Ly: $Ly"
    echo "Lz: $Lz"
    echo "delt_t: $delt_t"
    echo "t_final: $t_final"
    echo "dim: $dim"

    if [[ "$debug_mode" -eq 1 ]]; then
        echo "Launching MPI under gdb..."
        mpiexec -n 1 xterm -e gdb --args ./permafrost -initial_PFgeom -temp_initial
    else
        mpiexec -np 8 ./permafrost -initial_PFgeom -temp_initial -snes_rtol 1e-3 \
        -snes_stol 1e-6 -snes_max_it 7 -ksp_gmres_restart 150 -ksp_max_it 1000 \
        -ksp_converged_reason -snes_converged_reason -snes_linesearch_monitor \
        -snes_linesearch_type basic | tee "$folder/outp.txt"
    fi
}

################################################################################
# Copy relevant scripts to folder and save summary parameters to .dat and CSV
################################################################################
finalize_results() {
    echo "Finalizing results..."
    cd ./scripts
    cp run_permafrost.sh plotpermafrost.py plotSSA.py plotPorosity.py $folder
    cd ../src
    cp permafrost.c $folder
    cd ../
    write_parameters_to_csv

    # Save simulation parameters
    cat << EOF > $folder/sim_params.dat
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
}

################################################################################
# Run post-processing plotting script
################################################################################
run_plotting() {
    echo "Queuing plotpermafrost.py"
    ./scripts/run_plotpermafrost.sh $name
}

################################################################################
# USER-DEFINED SIMULATION SETTINGS
################################################################################
echo " "
echo "Starting permafrost simulation workflow"
echo " "

debug_mode=0  # set to 1 to enable gdb debugging

# Enable debug mode via command-line argument
if [[ $# -gt 0 && $1 == "debug" ]]; then
    debug_mode=1
    echo "🛠️ Running in DEBUG mode"
fi

delt_t=1.0e-4
t_final=100*24*60*60
n_out=100
t_final=$(echo "$t_final" | bc -l)

if (( $(echo "$t_final <= 0" | bc -l) )); then
    echo "❌ Error: t_final must be > 0. Got $t_final"
    exit 1
fi

humidity=0.70
temp=-20.0
grad_temp0X=0.0
grad_temp0Y=3.0
grad_temp0Z=0.0
dim=2
filename="circle_data.csv"
title="RandomGrains"

compile_code

create_folder

set_parameters

finalize_results

run_simulation

run_plotting

echo "-------------------------------------------------------------------------"
echo " "
echo "✅ Done with permafrost simulation!"
echo "-------------------------------------------------------------------------"
echo " "
