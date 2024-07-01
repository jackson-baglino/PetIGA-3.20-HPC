import matplotlib.pyplot as plt
import numpy as np

# Read data from the file
filename = "/Users/jacksonbaglino/SimulationResults/DrySed_Metamorphism/" \
    "NASA_read/res_2023-12-12__22.19.37/SSA_evo.dat"
grainFile = "/Users/jacksonbaglino/PetIGA/demo/input/grainReadFile.csv"

try:
    # Use genfromtxt for more flexible data loading
    data = np.genfromtxt(filename, delimiter=' ', dtype=float, comments='#', filling_values=np.nan)
    # Drop rows with NaN values
    data = data[~np.isnan(data).any(axis=1)]
    ssa_data = data[200:-1, 0]
    porosity = data[200:-1, 1]
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    exit()
except Exception as e:
    print(f"Error reading data from '{filename}': {e}")
    exit()

print("Loaded data from SSA_evo.dat:")
# print(data)

try:
    # Assuming grainFile is a CSV file with columns for grain data and porosity
    grain_data = np.loadtxt(grainFile, delimiter=',', usecols=(2,))
except FileNotFoundError:
    print(f"Error: File '{grainFile}' not found.")
    exit()
except Exception as e:
    print(f"Error reading data from '{grainFile}': {e}")
    exit()

grain_data = grain_data * np.sqrt(2 * (2e-3)**2) / np.sqrt(2 * (200)**2)

SSA0 = np.sum(2 * np.pi * grain_data)
print(f"SSA0 = {SSA0}")

# Normalize SSA data by the third column of grain data
c = ssa_data[0] / SSA0
normalized_ssa_data = ssa_data / c / 10

# Compute initial porosity
domain = 2e-3*1.583e-3

porosity = porosity/domain

# Create a time array (assuming data points are equally spaced)
time = np.linspace(1, 3, len(normalized_ssa_data))




# Create the main figure and axis for Surface Area Evolution
plt.figure(figsize=(10, 6))
ax1 = plt.gca()  # Get current axis
line1, = ax1.plot(time, normalized_ssa_data, 'b-', label='Surface Area Evolution')  # 'b-' for blue line
ax1.set_xlabel('Time [days]', fontsize=18)
ax1.set_ylabel(r'Surface Area [m]', fontsize=18, color='b')  # Blue color for the left y-axis
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Evolution of Surface Area and Porosity', fontsize=24)
ax1.grid(False)

# Create the second axis for Porosity Evolution
ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis
line2, = ax2.plot(time, porosity, 'r-', label='Porosity Evolution')  # 'r-' for red line
ax2.set_ylabel('Porosity', fontsize=18, color='r')  # Red color for the right y-axis
ax2.tick_params(axis='y', labelcolor='r')

# Set font sizes for tick labels
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

# # Create a combined legend
# lines = [line1, line2]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper right')

# Save and show the plot
plt.savefig("combined_evolution_N-150_dual_y_axis.png")
plt.show()









# # Plotting the Surface Area data
# plt.figure(figsize=(10, 6))
# plt.plot(time, normalized_ssa_data, label='Surface Area Evolution')
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Surface Area', fontsize=18)
# plt.title('Surface Area Evolution', fontsize=24)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(True)
# plt.legend()
# plt.savefig("ssa_evolution_N-150.png")
# plt.show()

# # Plotting the second column from "filename"
# plt.figure(figsize=(10, 6))
# plt.plot(time, porosity, label='Porosity')
# plt.xlabel('Time [days]', fontsize=18)
# plt.ylabel('Porosity', fontsize=18)
# plt.title('Porosity Evolution', fontsize=24)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(True)
# plt.legend()
# plt.savefig("poroposity_evolution_N-150.png")
# plt.show()