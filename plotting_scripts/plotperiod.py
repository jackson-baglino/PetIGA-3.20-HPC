from igakit.io import PetIGA,VTK
from numpy import linspace
import glob
import os
nrb = PetIGA().read("igasolP.dat")
#uniform = lambda U: linspace(U[0], U[-1], 400)
for infile in glob.glob("solP*.dat"):
	name = infile.split(".")[0]
	number = name.split("l")[1]
	root = './solV'+number+'.vtk'
	if not os.path.isfile(root):
		sol = PetIGA().read_vec(infile,nrb)
		outfile = root 
		VTK().write(outfile,  
	            nrb,             
	            fields=sol,     
#	            sampler=uniform, 
    		    scalars = {'ice':0,'temp':1,'pres':2}) 
#		    gradients ={'presgrad':[2,2]})
#	    	    vectors = {'vel_darc':[4,5]})


nrbS = PetIGA().read("igasolS.dat")
#uniform = lambda U: linspace(U[0], U[-1], 400)
for infile in glob.glob("solS*.dat"):
	name = infile.split(".")[0]
	number = name.split("S")[1]
	root = './solS'+number+'.vtk'
	if not os.path.isfile(root):
		sol = PetIGA().read_vec(infile,nrbS)
		outfile = root 
		VTK().write(outfile,  
	            nrb,             
	            fields=sol,     
#	            sampler=uniform, 
    		    scalars = {'sed':0}) 
