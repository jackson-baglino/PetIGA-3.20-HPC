import numpy as np
import pyvista as pv

def read_dat_file(filename):
    # Read the .dat file and extract the points and faces
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Assuming the .dat file format is known and structured
    points = []
    faces = []
    for line in lines:
        values = line.split()
        if len(values) == 3:  # Assuming points have 3 values
            points.append([float(v) for v in values])
        elif len(values) > 3:  # Assuming faces have more than 3 values
            faces.append([int(v) for v in values])
    
    points = np.array(points)
    faces = np.array(faces)
    
    return points, faces

def convert_to_stl(input_file, output_file):
    points, faces = read_dat_file(input_file)
    
    # Create a pyvista PolyData object
    mesh = pv.PolyData(points, faces)
    
    # Save the mesh as an STL file
    mesh.save(output_file, binary=False)

# Usage
input_file = 'path/to/your/input.dat'
output_file = 'path/to/your/output.stl'

convert_to_stl(input_file, output_file)

