from PIL import Image

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rd
import shutil
import sys

"""
Created on Fri Oct 6

Usage:
This file is used to read in a data-set of spherical grains that gives the (1) 
grain number, (2) the type of measurement of the grain (e.g., diameter), (3) the
numerical measurement, and (4) the units of the measurement. The script then 
identifies the best-fit distributiion. The script will lastly generate a file 
that contains the centers and radii of N particles that have been sampled from 
the same distribution as the provided data set.

@author: Jackson Baglino
"""


# Check if the correct number of command-line arguments is provided
if len(sys.argv) == 1:
    # outDir = "/Users/jacksonbaglino/Documents/scorpion/Academics/Research/" \
    #     "PhaseFieldDEM/LSDEM_Fluid_2D/input/"
    # grainOut = "/Users/jacksonbaglino/Documents/scorpion/Academics/Research/" \
    #     "PhaseFieldDEM/LSDEM_Fluid_2D/input/"

    outDir          = "/Users/jacksonbaglino/LSDEM/LSDEM_Fluid_2D/input/"
    grainOut        = "/Users/jacksonbaglino/LSDEM/LSDEM_Fluid_2D/input/"
    
elif len(sys.argv) != 3:
    print("Usage: python my_python_program.py <input> <outDir>")
    sys.exit(1)
else:
    # Retrieve the "input" and "outDir" variables from the command-line args
    outDir = sys.argv[1]
    grainOut = sys.argv[2]

print("Output directory:", outDir)
print("grainOut directory:", grainOut)

inFile = 'fitParams.dat'
inFilePath = outDir + inFile

fitParams = []

N =         20
Lx =        4e-3
# Lx =        2e-3
Ly =        2e-3
Nx =        400
# Nx =        200
Ny =        200
numPoints = 55

ratio = Nx/Lx


# Grain properties 
rho = 0.10
kn = 6e6 #Eice=6 GPa
ks = 5.4e6
mu = 0.99  #(Between 0.7 and 0.8)
eps = 0.5

dx = Lx/Nx
dy = Ly/Ny

kernel_size = (3, 3)
sigma = 2

plotFlag =  1
maxIter =   1e6

centX =     np.zeros((N,1))
centY =     np.zeros((N,1))
radius =    np.zeros((N, 1))

padding = 15
theta = np.linspace(0, 2*np.pi, numPoints)

np.random.seed(10)


# Define functions
def readData(inFilePath):
    # Read input parameters
    with open(inFilePath, 'r') as file:
        lines = file.readlines()  # Read all lines into a list
        num_lines = len(lines)
        
        # Check if there are at least 2 lines for minDiam and maxDiam
        if num_lines >= 2:
            minDiam = float(lines[-2].strip())  # Penultimate line
            maxDiam = float(lines[-1].strip())  # Last line

            # Extract the lines corresponding to fitParams
            fitParamLines = lines[0:num_lines - 2]

            # Convert fitParams lines to float values and store in fitParams
            fitParams = [float(line.strip()) for line in fitParamLines]
        
    print("Fit Parameters:",    fitParams)
    print("minDiam:",           minDiam)
    print("maxDiam:",           maxDiam)

    return fitParams, minDiam, maxDiam


def initializeParticles(inFilePath):
        # Initialize parameters for randomly generating particles, where their 
    # diameter was sampled from a Beta distribution.
    n_act = 0
    counts = 0
    i = 0

    # Read in data
    fitParams, minDiam, maxDiam = readData(inFilePath)

    print("Intiailizing Particles!")

    while counts < N*maxIter:
        '''
        Get random values for the centroid and the radius
        Sample the radius from the beta distribution and the centroids from a 
        uniform distribution.
        '''
        diam = np.random.beta(fitParams[0],fitParams[1])
        diam = (diam*fitParams[3] + fitParams[2])*1e-6

        # Ensure the random diamter is within our desired range
        while diam < minDiam and diam > maxDiam:
            diam = np.random.beta(fitParams[0],fitParams[1])
            diam = (diam*fitParams[3] + fitParams[2])*1e-6

        # Define rc as the radius. Note that the data we read in were the 
        # diameters, however, we want to write the radii. It would be a good 
        # idea in the future the functionalize this code such that it executes 
        # this one line if the given data are diameters, and do something 
        # different for radii.
        rc = diam/2

        # Randomly generate centroid
        # NOTE: We have are allowing particles to be generated at places 3*Ly 
        #       such that they can fall in a more realistic manner.
        xc = np.random.uniform(rc,Lx-rc)
        yc = np.random.uniform(rc,Ly-rc)

        # Ensure the particle is found in our domain
        flag = 1

        if xc-1.1*rc < 0:
            flag = 0
        elif yc+1.1*rc > Ly:      # Note that we allow particles to start 3x
                                # higher than Ly. This is so we can allow for a 
                                # more accuare pluvation experiment.
            flag = 0
        elif xc+1.1*rc > Lx:
            flag = 0
        elif yc-1.1*rc < 0:
            flag = 0

        # Ensure particles do not overlap
        if flag == 1:
            for j in range(n_act):
                dist = np.sqrt(np.square(xc-centX[j]) + np.square(yc-centY[j]))
                if dist < 1.1*(rc+radius[j]):
                    flag = 0
        
        # If flag == 1, write the particle information to the propery variables
        if flag == 1:
            print(f"New particle ({n_act+1}) initialized in {i+1} iterations!!")

            centX[n_act] = xc
            centY[n_act] = yc
            radius[n_act] = rc

            n_act += 1
            i = 0

        # Stop the loop if we have initialized all our particles
        if n_act == N:
            print(f"{n_act} grains in {counts+1} iterations.\n")
            counts = N*maxIter+1
        else:
            counts += 1
            i += 1

        if counts == N*maxIter:
            print("FAILED TO INITIALIZE GRAINS!!")
            return 0

    for i in range(N):
        print(f"xc = {centX[i][0]:.3e},    yc = {centY[i][0]:.3e},    " \
              f"rad = {radius[i][0]:.3e}.")
        

    # Plot circles as scatter plot points
    if plotFlag == 1:
        plt.figure()
        for i in range(N):
            circle = plt.Circle((centX[i], centY[i]), radius[i], fill=True, 
                                color='blue', linewidth=0.25)

            plt.gca().add_patch(circle)
            
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, Lx)
        plt.ylim(0, Ly)
        plt.title("Particles")

        plt.figure()
        plt.scatter(centX, centY, marker='o', s=50)
        plt.show()

    return centX, centY, radius

def smoothDirac(phi, eps):
    '''
    Returns smoothed dirac delta function
    '''
    cond = (np.absolute(phi) <= eps).astype(int)
    res = cond * 0.5/eps * (1 + np.cos(np.pi*phi/eps))

    return res


def smoothHeaviside(phi, eps):
    '''
    Returns smoothed Heaviside step function
    '''    
    cond1 = (np.absolute(phi) <= eps).astype(int)
    cond2 = (phi > eps).astype(int)
    res = cond1 * 0.5 * (1 + phi/eps + np.sin(np.pi*phi/eps)/np.pi) + cond2
    
    return res


# Define main
def main():
    # Initialize the particles
    centX, centY, radius = initializeParticles(inFilePath)

    # Create mesh grids for x and y coordinates
    x, y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))

    # Create an empty image
    img0 = np.zeros((Ny, Nx), dtype=np.uint8)

    for idx in range(N):
        print(f"Drawing particle {idx + 1} in image.")
        
        dist = np.sqrt((x - centX[idx])**2 + (y - centY[idx])**2)

        # Set all points within the circle to 1
        img0[dist < radius[idx]] = 255

    if plotFlag == 1:
        f1 = plt.figure()
        plt.imshow(img0, cmap='jet')
        plt.title("Image file")
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()

    img = Image.fromarray(img0)

    outFile = outDir + "img0.jpg"
    img.save(outFile)
    # print(f"Image saved as {outFile}.")

    # Now we want to generate our level-sets
    # Get a list of all files that begin with 'grainproperty'
    grain_files = glob.glob(os.path.join(grainOut, 'grainproperty*'))

    # Iterate over the list and remove each file
    for file in grain_files:
        if os.path.isfile(file):
            os.remove(file)
        
    if not os.path.exists(grainOut):
        os.mkdir(grainOut)

    # Set up file for writing positions
    g = open(grainOut + "positions.dat", 'w')
    h = open(grainOut + "morphIDs.dat", 'w')
    k = open(grainOut + "rads.dat", 'w')

    centX = centX/dx
    centY = centY/dy

    for i in range(N):
        # Set up file for writing
        f = open(grainOut + 'grainproperty' + str(i+1) + '.dat','w')  #str i-1

        boxDiam = 2.2*radius[i]
        dim = int(np.ceil(boxDiam / max(Lx, Ly) * max(Nx, Ny) + 2 * padding))
        print(f"Particle {i+1} has a level-set array of size ({dim}, {dim}).")

        # Initialize the level set
        phi0 = np.zeros((dim, dim))

        xspan = np.linspace(-dim/2*dx, dim/2*dx, dim)
        yspan = np.linspace(-dim/2, dim/2, dim)*dy

        XX, YY = np.meshgrid(xspan, yspan)

        distField = np.sqrt(np.square(XX) + np.square(YY))
        phi0 = distField - (radius[i])

        phi0 = phi0/dx

        noise = np.random.uniform(-1, 1, numPoints)/ratio

        noisey_radius = radius[i] + 2000*noise*radius[i]
        # noisey_radius = radius[i]

        # xp = (noisey_radius * np.cos(theta))/dx
        # yp = (noisey_radius * np.sin(theta))/dy

        xp = np.multiply(noisey_radius, np.cos(theta))/dx
        yp = np.multiply(noisey_radius, np.sin(theta))/dy

        bpointsCM = np.column_stack((xp, yp))

        # Display the level-set arrays
        if N < 4:
            plt.figure()
            plt.imshow(phi0, cmap='twilight_shifted')
            plt.colorbar()

            plt.plot(xp + dim/2, yp + dim/2, 'ro', markersize=5)
            plt.plot(xp + dim/2, yp + dim/2, 'b-', linewidth=1)

            plt.title("Level Set image file")
            plt.gca().invert_yaxis()
            plt.show()
        elif N - i < 4:
            plt.figure()
            plt.imshow(phi0, cmap='twilight_shifted')
            plt.colorbar()

            plt.plot(xp + dim/2, yp + dim/2, 'ro', markersize=5)
            plt.plot(xp + dim/2, yp + dim/2, 'b-', linewidth=1)

            plt.title("Level Set image file")
            plt.gca().invert_yaxis()
            plt.show()

        # Determine the mass from the level set
        m = rho * smoothHeaviside(-phi0, eps).ravel().sum()
        print(f"The mass is {m}.")

        # Compute center of mass
        (ny, nx) = phi0.shape
        X = np.arange(nx)
        Y = np.arange(ny)

        cx = rho / m * np.dot(smoothHeaviside(-phi0, eps), X).sum()
        cy = rho / m * np.dot(smoothHeaviside(-phi0, eps).T, Y).sum()

        # Compute moment of inertia
        x_dev = np.power(X-cx,2)
        y_dev = np.power(Y-cy,2)

        I = rho * np.dot(smoothHeaviside(-phi0, eps), x_dev).sum() + \
            rho * np.dot(smoothHeaviside(-phi0, eps).T, y_dev).sum()
        print(f"The MOI is {I}.")

        # Compute bbox radius
        bbox = np.linalg.norm(bpointsCM, axis = 1).max()
        print(f"bbox = {bbox}.")

        # Write positions and morphIDs
        g.write('%5.3f'% centX[i] + ' ' + '%5.3f'% centY[i] + ' 0.0' + '\n')
        h.write('%d'%(i+1) + '\n')

        k.write('%9.8f'% radius[i] + '\n')

        # Write properties to grainProp file
        f.write('%5.3f'% m + '\n')                           # mass
        f.write('%5.3f'% I*1000 + '\n')                           # moment of inertia
        f.write('%5.3f' % cx + ' ' + '%5.3f'% cy + '\n')     # center of mass
        f.write('%d' % numPoints + '\n')                     # # boundary points
        f.write(" ".join('%5.6f' % x for x in bpointsCM.ravel().tolist())+'\n') 
        f.write('%5.3f'% bbox + '\n')                      # bounding box radius
        f.write('%5.3f' % nx + ' ' + '%5.3f'% ny + '\n')     # lset dims
        f.write(" ".join(str(x) for x in phi0.ravel().tolist()) + '\n') 
        f.write('%5.3f'% kn + '\n')                           # kn
        f.write('%5.3f'% ks + '\n')                           # ks
        f.write('%5.3f'% mu)                                  # mu
        f.close()

    g.close()
    h.close()
    k.close()

    return 0

if __name__ == "__main__":
    plt.ion()
    main()
    plt.pause(60)
    plt.ioff()