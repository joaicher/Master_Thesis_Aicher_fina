#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:41:45 2021

@author: konstantinos
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('../stat-mech')
from utilities import getSamplingParameters, convertMatrixToTensor, convertTensorToMatrix

# Graph to plot
N = 8 
Z = 6.5
lMu = 40
lZ = [1,0.1,10,0.1]
r = 10
rho = 0.1

# Find corresponding idx in lZList file
lZList, _, nRealizations = getSamplingParameters(N)
lZList = np.array(lZList)
idx = np.where(np.all(lZList == lZ, axis=1))[0]
# Find corresponding idx in elastic stiffness file
idxC = idx*nRealizations+r

# Read elastic stiffness
filename = "../data/N="+str(N)+"-C/N="+str(N)+"_Z="+str(Z)+"_lMu="+str(lMu)+"_rho="+str(rho)+"-elasticity.csv"
raw_data = pd.read_csv(filename, skipinitialspace=True, skiprows=0)

# Function to get directional C
def directional_stiffness(C, alphas):
    stiffness = []
    for alpha in alphas:
        director = np.array([[np.cos(alpha), np.sin(alpha)]])
        directormatrix = director *  director.transpose()
        dV = np.array([[directormatrix[0,0], directormatrix[1,1],  directormatrix[0,1]]])
        stiffness.append(1./dV.dot(np.linalg.inv(C).dot(dV.transpose()))[0,0])
    return stiffness

# Plot
dtheta = 0.02
theta = np.arange(0, 2*np.pi+dtheta, dtheta)
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(polar=True)
C = raw_data.iloc[idxC,2:].to_numpy().reshape((3,3))

ax.plot(theta, directional_stiffness(C, theta))
ax.fill_between(theta, 0, directional_stiffness(C, theta), alpha=0.2)
ax.set_yticklabels([])
ax.tick_params(axis='x', which='major', labelsize=14, pad=5) 
plt.tight_layout()
lZ0, lZ45, lZ90, lZ135 = lZ
latticeStr = '/N=' + str(N) + '_Z=' + str(Z)
latticeStr += '_lZ0=' + str(lZ0) + '_lZ1=' + str(lZ45)
latticeStr += '_lZ2=' + str(lZ90) + '_lZ3=' + str(lZ135)
latticeStr += '_lMu=' + str(lMu) + '_r=' + str(r)
#fig.savefig('../figures/directional-stiffness/' + latticeStr + '.png')
plt.show()
