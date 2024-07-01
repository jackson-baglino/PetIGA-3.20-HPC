#!/bin/zsh

if [[ -n $1 ]]; then
    echo "Copying run_plotNASA.py to: $1"
    dir=/Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/NASA_read/$1
    cp plotNASA.py $dir
    cp writeNASA2CSV.py $dir

    exec_dir=/Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/NASA_read/$1/

    echo " "
    echo $exec_dir
    echo " "

    cd $exec_dir
    echo " "
    echo "Directory changed"
    echo " "
    echo "Working directory: $(pwd)"
    echo " "
    echo "Executing python script"
    echo " "

    python3.11 ~/SimulationResults/DrySed_Metamorphism/NASA_read/$1/plotNASA.py
    python3.11 ~/SimulationResults/DrySed_Metamorphism/NASA_read/$1/writeNASA2CSV.py
else
    echo "No inputs are given. Assume we are already in the results folder"
    echo " "
    python3.11 plotNASA.py
    python3.11 writeNASA2CSV.py
fi

