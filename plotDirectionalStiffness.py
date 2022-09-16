#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: konstantinos
"""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import linalg as LA
import torch.nn as nn

rew_loss = nn.MSELoss()
def compute_loss(stiffness_fem, stiffness_goal):
    return rew_loss(stiffness_goal, stiffness_fem)


# Read elastic stiffness from file
# filename = "C.dat"
# C = np.loadtxt(filename)
# or provide it here
# C = np.array([[0.3017, -0.00034, -0.00045],
#               [-0.00034, 0.01214, 0.00039],
#               [-0.00045, 0.00039, 0.00146]])
stiffness_goal = torch.tensor([200, 0, 0, 0, 100, 0, 0, 0, 30.7692])
stiffness_goal = stiffness_goal / LA.vector_norm(stiffness_goal)
print(stiffness_goal)
C = np.array([[0.8861, 0, 0],
              [0, 0.4430, 0.000],
              [0.000, 0.000, 0.1363]])
# 4x4: [200, -2.46519e-32, 0, 6.16298e-33, 66.6667, 2.22045e-16, 1.44855e-17, 1.22931e-16, 22.2222]
stiffness_goal_4x = torch.tensor([200, -2.46519e-32, 0, 6.16298e-33, 66.6667, 2.22045e-16, 1.44855e-17, 1.22931e-16, 22.2222])
stiffness_goal_4x = stiffness_goal_4x / LA.vector_norm(stiffness_goal_4x)
stiffness_goal_array = np.array([[stiffness_goal_4x[0], stiffness_goal_4x[1], stiffness_goal_4x[2]],
                            [stiffness_goal_4x[3], stiffness_goal_4x[4], stiffness_goal_4x[5]],
                            [stiffness_goal_4x[6], stiffness_goal_4x[7], stiffness_goal_4x[8]]])

# 4x4: stiffness_tensor = torch.tensor([137.018, 0.153163, -3.70466, 0.153163, 35.7241, -3.22466, -3.70466, -3.22466, 19.9002])
stiffness_tensor = torch.tensor([278.345, 11.5108, -14.7381, 11.5108, 65.7114, -17.8475, -14.7381, -17.8475, 51.2944])
stiffness_tensor = stiffness_tensor / LA.vector_norm(stiffness_tensor)

print(compute_loss(stiffness_tensor, stiffness_goal_4x))
# stiffness_array is just the stiffness tensor as 3x3 matrix
stiffness_array = np.array([[stiffness_tensor[0], stiffness_tensor[1], stiffness_tensor[2]],
                            [stiffness_tensor[3], stiffness_tensor[4], stiffness_tensor[5]],
                            [stiffness_tensor[6], stiffness_tensor[7], stiffness_tensor[8]]])

C2 = np.array([[0.0017, -0.00034, -0.00045],
              [-0.00034, 0.8214, 0.00039],
              [-0.00045, 0.00039, 0.00146]])

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
ax.plot(theta, directional_stiffness(stiffness_goal_array, theta))
ax.plot(theta, directional_stiffness(stiffness_array, theta))
ax.fill_between(theta, 0, directional_stiffness(stiffness_goal_array, theta), alpha=0.2)
ax.set_yticklabels([])
ax.tick_params(axis='x', which='major', labelsize=14, pad=5) 
plt.tight_layout()
plt.savefig("directional_stiffness")
plt.show()
