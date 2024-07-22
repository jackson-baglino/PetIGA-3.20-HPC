import numpy as np
import glob
import os
from igakit.io import PetIGA, VTK
from numpy import linspace
from skimage import measure
import pyvista as pv
import vtk

from rich.traceback import install
install()

def create_vtk_files():
    maxNum = 300
    numX = int(os.getenv("Nx", maxNum))
    numY = int(os.getenv("Ny", maxNum))
    numZ = int(os.getenv("Nz", maxNum))

    num_points = max(numX, numY, numZ)
    print(f"Number of points: {num_points}.\n")

    nrb = PetIGA().read("igasol.dat")
    uniform = lambda U: linspace(U[0], U[-1], num_points)

    for infile in glob.glob("sol*.dat"):
        name = infile.split(".")[0]
        number = name.split("l")[1]
        vtk_root = f'./vtkOut/solV_{number}.vtk'
        # stl_root = f'./stlOut/IcePhase_{number}.stl'

        sol = PetIGA().read_vec(infile, nrb)
        if os.path.exists(vtk_root):
            os.remove(vtk_root)

        VTK().write(vtk_root,
                    nrb, fields=sol, 
                    sampler=uniform, 
                    scalars={'IcePhase': 0, 'Temperature': 1, 'VaporDensity': 2})
        print(f"Created: {vtk_root}.\n")

        verify_vtk_fields(vtk_root)
        # convert_vtk_to_stl(vtk_root, stl_root)
        # print(f"Converted: {vtk_root} to {stl_root}.\n")

# def convert_vtk_to_stl(vtk_file, stl_file):
#     mesh = pv.read(vtk_file)

#     if 'IcePhase' in mesh.array_names:
#         ice_phase = mesh['IcePhase']
#         mesh.point_data['scalars'] = ice_phase
#         mesh.set_active_scalars('scalars')

#         threshold = 0.5
#         thresholded_mesh = mesh.threshold(threshold, scalars='scalars')

#         poly_data = thresholded_mesh.extract_surface()
#         smoothed_mesh = poly_data.smooth(n_iter=50, relaxation_factor=0.1)

#         print(f"Writing: {stl_file}....................", end="")
#         smoothed_mesh.save(stl_file)
#         print(" Complete!\n")
#     else:
#         print(f"Error: 'IcePhase' field not found in {vtk_file}")

def verify_vtk_fields(vtk_file):
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    vtk_data = reader.GetOutput()

    cell_data = vtk_data.GetCellData()
    point_data = vtk_data.GetPointData()
    
    print(f"Fields in {vtk_file}:")
    print("Cell Data Fields:")
    for i in range(cell_data.GetNumberOfArrays()):
        print(f"  {cell_data.GetArrayName(i)}")
        
    print("Point Data Fields:")
    for i in range(point_data.GetNumberOfArrays()):
        print(f"  {point_data.GetArrayName(i)}")

def apply_contour_filter(vtk_data, field_name, threshold_value, vtk_file):
    if field_name not in [vtk_data.GetPointData().GetArrayName(i) for i in range(vtk_data.GetPointData().GetNumberOfArrays())]:
        print(f"Field '{field_name}' not found in the VTK file.")
        verify_vtk_fields(vtk_file)
        return None

    print("Applying contour filter...")
    contour_filter = vtk.vtkContourFilter()
    contour_filter.SetInputData(vtk_data)
    contour_filter.SetValue(0, threshold_value)
    contour_filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
    contour_filter.Update()
    return contour_filter.GetOutput()

def save_vtk_file(vtk_data, output_filename):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(vtk_data)
    writer.Write()

def process_files(input_folder, output_folder, field_name, threshold_value):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    vtk_files = glob.glob(os.path.join(input_folder, '*.vtk'))
    
    for vtk_file in vtk_files:
        print(f'Processing {vtk_file}...')
        reader = vtk.vtkStructuredGridReader()
        reader.SetFileName(vtk_file)
        reader.Update()
        vtk_data = reader.GetOutput()

        contour_data = apply_contour_filter(vtk_data, field_name, threshold_value, vtk_file)
        
        if contour_data is not None:
            output_filename = os.path.join(output_folder, os.path.basename(vtk_file).replace('.vtk', '_contour.vtp'))
            save_vtk_file(contour_data, output_filename)
            print(f'Saved contour to {output_filename}')
        else:
            print(f"Error processing {vtk_file}")

if __name__ == '__main__':
    input_folder = './vtkOut/'
    output_folder = './stlOut/'
    field_name = 'IcePhase'
    threshold_value = 0.5
    
    create_vtk_files()
    process_files(input_folder, output_folder, field_name, threshold_value)












# import numpy as np
# import glob
# import os
# from igakit.io import PetIGA, VTK
# from numpy import linspace
# import pyvista as pv
# import vtk

# from rich.traceback import install
# install()

# def create_vtk_files():
#     """
#     Creates VTK files for ice and sediment grains and converts them to STL files.

#     Reads input data files and generates VTK files for visualization.
#     The VTK files contain information about the ice and sediment grains,
#     including ice phase, temperature, vapor density, and sediment phase.
#     """

#     # Get the values of Nx, Ny, Nz from environment variables
#     maxNum = 200
#     numX = int(os.getenv("Nx", maxNum))
#     numY = int(os.getenv("Ny", maxNum))
#     numZ = int(os.getenv("Nz", maxNum))

#     num_points = max(numX, numY, numZ)

#     # Print the number of points
#     print(f"Number of points: {num_points}.\n")

#     # Read the input data file for ice grains
#     nrb = PetIGA().read("igasol.dat")

#     # Define a uniform sampling function
#     uniform = lambda U: linspace(U[0], U[-1], num_points)

#     # Import ice grains:
#     for infile in glob.glob("sol*.dat"):
#         name = infile.split(".")[0]
#         number = name.split("l")[1]
#         vtk_root = f'./vtkOut/solV_{number}.vtk'

#         # Read the solution vector from the input file
#         sol = PetIGA().read_vec(infile, nrb)

#         # Remove the existing VTK file if it exists
#         if os.path.exists(vtk_root):
#             os.remove(vtk_root)

#         # Write the VTK file with ice grain information
#         VTK().write(vtk_root,
#                     nrb, fields=sol, 
#                     sampler=uniform, 
#                     scalars={'IcePhase': 0, 'Temperature': 1, 'VaporDensity': 2})
#         print(f"Created: {vtk_root}.\n")

# def verify_vtk_fields(vtk_file):
#     """
#     Verifies and lists all fields in the VTK file.

#     Parameters:
#     vtk_file (str): Path to the VTK file.
#     """
#     reader = vtk.vtkStructuredGridReader()
#     reader.SetFileName(vtk_file)
#     reader.Update()
#     vtk_data = reader.GetOutput()

#     cell_data = vtk_data.GetCellData()
#     point_data = vtk_data.GetPointData()
    
#     print(f"Fields in {vtk_file}:")

#     print("Cell Data Fields:")
#     for i in range(cell_data.GetNumberOfArrays()):
#         print(f"  {cell_data.GetArrayName(i)}")
        
#     print("Point Data Fields:")
#     for i in range(point_data.GetNumberOfArrays()):
#         print(f"  {point_data.GetArrayName(i)}")

# def apply_contour_filter(vtk_data, field_name, threshold_value, vtk_file):
#     """
#     Applies a contour filter to the given VTK data.

#     Parameters:
#     vtk_data: The input VTK data.
#     field_name (str): The name of the field to contour.
#     threshold_value (float): The contour threshold value.
#     vtk_file (str): Path to the VTK file.

#     Returns:
#     vtkPolyData: The contoured data.
#     """
#     # if field_name not in [vtk_data.GetPointData().GetArrayName(i) for i in range(vtk_data.GetPointData().GetNumberOfArrays())]:
#     #     print(f"Field '{field_name}' not found in the VTK file.")
#     #     verify_vtk_fields(vtk_file)
#     #     return None

#     contour_filter = vtk.vtkContourFilter()
#     contour_filter.SetInputData(vtk_data)
#     contour_filter.SetValue(0, threshold_value)
#     contour_filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field_name)
#     contour_filter.Update()
#     return contour_filter.GetOutput()

# def save_vtk_file(vtk_data, output_filename):
#     """
#     Saves the given VTK data to a file.

#     Parameters:
#     vtk_data: The VTK data to save.
#     output_filename (str): The path to the output file.
#     """
#     writer = vtk.vtkXMLPolyDataWriter()
#     writer.SetFileName(output_filename)
#     writer.SetInputData(vtk_data)
#     writer.Write()

# def process_files(input_folder, output_folder, field_name, threshold_value):
#     """
#     Processes VTK files by applying a contour filter and saving the results.

#     Parameters:
#     input_folder (str): The path to the input VTK files.
#     output_folder (str): The path to save the output files.
#     field_name (str): The name of the field to contour.
#     threshold_value (float): The contour threshold value.
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     vtk_files = glob.glob(os.path.join(input_folder, '*.vtk'))
    
#     for vtk_file in vtk_files:
#         print(f'Processing {vtk_file}...')
#         reader = vtk.vtkStructuredGridReader()
#         reader.SetFileName(vtk_file)
#         reader.Update()
#         vtk_data = reader.GetOutput()

#         contour_data = apply_contour_filter(vtk_data, field_name, threshold_value, vtk_file)
        
#         if contour_data is not None:
#             output_filename = os.path.join(output_folder, os.path.basename(vtk_file).replace('.vtk', '_contour.vtp'))
#             save_vtk_file(contour_data, output_filename)
#             print(f'Saved contour to {output_filename}')

# if __name__ == '__main__':
#     input_folder = './vtkOut/'
#     output_folder = './stlOut/'
#     field_name = 'IcePhase'  # Use the correct field name from the VTK files
#     threshold_value = 0.5
    
#     # Create VTK files
#     create_vtk_files()
    
#     # Process the created VTK files
#     process_files(input_folder, output_folder, field_name, threshold_value)
