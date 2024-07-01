#!/bin/zsh

echo " "
echo "Starting conversion..."

dir="/Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/NASAv2/res_2024-06-20__09.53.35"

# Create output directory
mkdir -p "$dir/stlOut"

inputDir=$dir"/vtkOut"
outputDir=$dir"/stlOut"

python3.11 plotNASA.py $inputDir $outputDir