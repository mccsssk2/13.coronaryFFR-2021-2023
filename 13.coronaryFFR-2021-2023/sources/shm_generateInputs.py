#!/usr/bin/env python
'''
-----
SRK, SHM. June 29, 2022.
A python program to generate N input vectors, one for each instance. The design sampling is uniform, the test sampling is normal around a given mean input vector.
Input: an integer N, for number of instances.
'''

## Imports
# standard
import fnmatch,os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import sys
import math
from scipy.signal import argrelextrema
# for LHS design.
from pyDOE import *
from scipy.stats.distributions import norm
from smt.sampling_methods import LHS
#
#
# get the uniformly distributed design vector for N instances.
N 				= int(sys.argv[1])  # number of instances
Num_inputs 		= 25
design 			= lhs(Num_inputs, samples=N)
# standard deviation in case generating test data.
stddev 			= 0.000001

# compliance range obtained from: SystemicResistance_ComplianceValuesTable2.pdf

# hypermia of 0.75 caused model failure. Reducing to 0.6 for a few days. July 6, 2022.
# sept 22, 2022. inputs 1 and 3 are now calculated using inputs 22, do not perturb inputs 1 and 3 here.
# Sept 26 2022. input23 is DoS perturbation of FFR1, 24 is FFR2, 25 is FFR3.
# all inputs are in range of +/-50%.
# these input values change as you construct the model. Take the constructed values.
whichCase=int(sys.argv[2]) # 0: all perturned. 1: HCT constant. 2: no DoS uncertainty. 3: total hypermia. 4: constant viscosity. case 4 is the same as case 1. do not do case 4.
if whichCase==0: # all perturbed.
	inputs        =[0.04, 100,1.06,0.83,0.6,0.8,0.025,10.0,6666.1,1.15e-3,2.6,3e6,-9, 2.0e7,-22.5,3.4e5,8.65e5,0.1,1.5,0.5,0.001,0.45,1.0, 1.0,1.0]
	inputs_min=[0.04, 50,1.06,0.83,0.3,0.8,0.025,10.0,3333.5,0.57e-3,2.6, 1.5e6,-14, 1.0e7,-33.5,1.7e5,4.3e5,0.1,0.75,0.25,0.001,0.20, 0.5, 0.5, 0.5]
	inputs_max=[0.04,150,1.06,0.83,0.9,0.8,0.025,10.0,9999.15,1.73e-3,2.6,4.5e6,-4.5, 3.0e7,-11.,5.1e5,13.0e5,0.1,2.25,.75,0.001,0.68, 1.0,1.0,1.0]
elif whichCase==1: # hematocrit constant.
	inputs        =[0.04, 100,1.06,0.83,0.6,0.8,0.025,10.0,6666.1,1.15e-3,2.6,3e6,-9, 2.0e7,-22.5,3.4e5,8.65e5,0.1,1.5,0.5,0.001,0.45,1.0, 1.0,1.0]
	inputs_min=[0.04, 50,1.06,0.83,0.3,0.8,0.025,10.0,3333.5,0.57e-3,2.6, 1.5e6,-14, 1.0e7,-33.5,1.7e5,4.3e5,0.1,0.75,0.25,0.001,0.4, 0.5, 0.5, 0.5]
	inputs_max=[0.04,150,1.06,0.83,0.9,0.8,0.025,10.0,9999.15,1.73e-3,2.6,4.5e6,-4.5, 3.0e7,-11.,5.1e5,13.0e5,0.1,2.25,.75,0.001,0.5, 1.0,1.0,1.0]
elif whichCase==2: # DoS is accurate.
	inputs        =[0.04, 100,1.06,0.83,0.6,0.8,0.025,10.0,6666.1,1.15e-3,2.6,3e6,-9, 2.0e7,-22.5,3.4e5,8.65e5,0.1,1.5,0.5,0.001,0.45,1.0, 1.0,1.0]
	inputs_min=[0.04, 50,1.06,0.83,0.3,0.8,0.025,10.0,3333.5,0.57e-3,2.6, 1.5e6,-14, 1.0e7,-33.5,1.7e5,4.3e5,0.1,0.75,0.25,0.001,0.20, 0.9, 0.9, 0.9]
	inputs_max=[0.04,150,1.06,0.83,0.9,0.8,0.025,10.0,9999.15,1.73e-3,2.6,4.5e6,-4.5, 3.0e7,-11.,5.1e5,13.0e5,0.1,2.25,.75,0.001,0.68, 1.0,1.0,1.0]
elif whichCase==3: # total hypermia acheived.
	inputs        =[0.04, 100,1.06,0.83,0.6,0.8,0.025,10.0,6666.1,1.15e-3,2.6,3e6,-9, 2.0e7,-22.5,3.4e5,8.65e5,0.1,1.5,0.5,0.001,0.45,1.0, 1.0,1.0]
	inputs_min=[0.04, 50,1.06,0.83,0.5,0.8,0.025,10.0,3333.5,0.57e-3,2.6, 1.5e6,-14, 1.0e7,-33.5,1.7e5,4.3e5,0.1,0.75,0.25,0.001,0.20, 0.5, 0.5, 0.5]
	inputs_max=[0.04,150,1.06,0.83,0.7,0.8,0.025,10.0,9999.15,1.73e-3,2.6,4.5e6,-4.5, 3.0e7,-11.,5.1e5,13.0e5,0.1,2.25,.75,0.001,0.68, 1.0,1.0,1.0]
elif whichCase==4: # all three are almost constant.
	inputs        =[0.04, 100,1.06,0.83,0.6,0.8,0.025,10.0,6666.1,1.15e-3,2.6,3e6,-9, 2.0e7,-22.5,3.4e5,8.65e5,0.1,1.5,0.5,0.001,0.45,1.0, 1.0,1.0]
	inputs_min=[0.04, 50,1.06,0.83,0.5,0.8,0.025,10.0,3333.5,0.57e-3,2.6, 1.5e6,-14, 1.0e7,-33.5,1.7e5,4.3e5,0.1,0.75,0.25,0.001,0.40, 0.9, 0.9, 0.9]
	inputs_max=[0.04,150,1.06,0.83,0.7,0.8,0.025,10.0,9999.15,1.73e-3,2.6,4.5e6,-4.5, 3.0e7,-11.,5.1e5,13.0e5,0.1,2.25,.75,0.001,0.50, 1.0,1.0,1.0]
else:
	sys.exit()

for i in range(Num_inputs):
	if inputs_min[i] > inputs[i] or inputs[i] > inputs_max[i]:
		print('input = ', i)
		sys.exit('bad inputs. try again. input at some input.')

# min and max approximation from mean (inputs) and stddev
# For a uniform sample, we define inputs_max and inputs_min explicitly as above.
# inputs_min = [inputs[j] - (stddev*math.sqrt(N-1)) for j in range(Num_inputs)] # min = mean - (stddev * (N-1)^1/2)
# inputs_max = [inputs[j] + (stddev*math.sqrt(N-1)) for j in range(Num_inputs)] # max = mean + (stddev * (N-1)^1/2)

'''# design data is uniformly sampled between min and max.
# Is this gaurenteed to be a LHS sample? We may need to randomly shuffle each column of design.
for i in range(Num_inputs):
	design[:, i] = np.random.uniform(low=inputs_min[i], high=inputs_max[i], size=N) # uniform sample.
	np.random.shuffle(design[:, i]) # random shuffle'''

## LHS sampling (limits)
for i in range(Num_inputs):
	xlimits = np.array([[0.0, N], [inputs_min[i], inputs_max[i]]]) # interval of the domain in each data dimension.
	design[:, i] = LHS(xlimits=xlimits)(N)[:, 1]

# the first instance must always be the mean, i.e. inputs vector.
design[0, :] = inputs

# write to file. 
# np.savetxt('Ninputs.dat', design, fmt="%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f")
np.savetxt('Ninputs.dat', design, fmt="%10.10f")
