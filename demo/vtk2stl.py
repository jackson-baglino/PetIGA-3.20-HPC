import vtk
import os
import sys

# Function to read .vtk file
def read_vtk(file_path):
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

# Function to apply a contour filter to extract the ice field surface
def contour_ice_field(data, contour_value=0.5):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(data)
    contour.SetValue(0, contour_value)
    contour.Update()
    return contour.GetOutput()

# Function to write the surface to .stl file
def write_stl(surface, output_file_path):
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(output_file_path)
    stl_writer.SetInputData(surface)
    stl_writer.Write()

# Main function to process all .vtk files
def process_vtk_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(input_directory):
      if file_name.endswith('.vtk'):
        input_file_path = os.path.join(input_directory, file_name)
        output_file_name = file_name.replace('.vtk', '.stl')
        output_file_path = os.path.join(output_directory, output_file_name)

        # Read the .vtk file
        data = read_vtk(input_file_path)

        # Apply contour filter to extract ice field surface
        surface = contour_ice_field(data)

        # Write to .stl file
        write_stl(surface, output_file_path)
        print(f"Processed {file_name} and saved as {output_file_name}")
        break

# Define input and output directories
input_directory = sys.argv[1]
output_directory = sys.argv[2]

# Process all .vtk files
process_vtk_files(input_directory, output_directory)

