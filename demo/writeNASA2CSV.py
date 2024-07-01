import numpy as np
import glob
import os
from igakit.io import PetIGA
from numpy import linspace
# import pandas as pd
import csv
from PIL import Image
from matplotlib import pyplot
from matplotlib import image as img

nrb = PetIGA().read("igasol.dat")
uniform = lambda U: linspace(U[0], U[-1], 1080)

# Create a list to store the field names
# field_names = ['IcePhase', 'Temperature', 'VaporDensity']
field_names = ['IcePhase']

# Create a directory for CSV files if it doesn't exist
csv_dir = './csv_output/'
os.makedirs(csv_dir, exist_ok=True)

# Import ice grains:
for infile in glob.glob("sol*.dat"):
    name = infile.split(".")[0]
    number = name.split("l")[1]
    root = os.path.join(csv_dir, 'solV' + number + '.csv')

    sol = PetIGA().read_vec(infile, nrb)
    
    with open(root, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # csv_writer.writerow(field_names)  # Write the header
        
        for i in range(sol.shape[1]):
            row_data = sol[:, i]  # Extract the data for a single field
            csv_writer.writerow(row_data[:,0])

    
    # Save the csv file as a gray-scale .png file
    csvFile = "./csv_output/solV" + number + ".csv"
    myData = np.genfromtxt(csvFile, delimiter=',')
    img.imsave("csv_output/solV" + number + ".png", myData, cmap='gray')
    Image.open("csv_output/solV" + number + ".png").convert('L').save("csv_output/solV" + number + ".png")

    print("Created: " + root + ".\n")

nrb2 = None
if os.path.exists("igasoil.dat"):
    # Import sediment grains:
    nrb2 = PetIGA().read("igasoil.dat")

    # Create a list to store the field names
    field_names = ['SedPhase']

    for infile in glob.glob("soil*.dat"):
        name = infile.split(".")[0]
        number = name.split("l")[1]
        root = os.path.join(csv_dir, 'soilV' + number + '.csv')

        sol = PetIGA().read_vec(infile, nrb2)
        
        with open(root, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(field_names)  # Write the header
            
            for i in range(sol.shape[1]):
                row_data = sol[:, i]  # Extract the data for a single field
                csv_writer.writerow(row_data)
                
        print("Created: " + root + ".\n")

