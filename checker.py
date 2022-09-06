# this file was created to check the results for a structure that is created in line 18-29; 

import gym
import gym.spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.nn.functional
from torch import linalg as LA
import time
import parameters as parameters

import fem as fem

import unitCell as unitCell
import get_stiffness_goal

a = unitCell.UnitCell(3)
a.bar_removed(5, 3)
#a.bar_removed(2,3)
a.bar_removed(4, 3)
#a.bar_removed(12, 3)
a.bar_removed(11, 3)
a.bar_removed(10, 3)
a.bar_removed(13,3)
#a.bar_removed(14,3)
#a.bar_removed(15, 3)
#random line
a.bar_removed(7, 3)
a.save('/Users/Johannes/Library/CloudStorage/OneDrive-Persönlich/Dokumente/ETH'
                                           '-Studium-Gesamt/MasterThesis/ae108-legacy/build/drivers/beamHomogenization/mesh' + str(2))
stiffness_fem, _ = fem.compute_rFEM(100, steps_counter=3, worker_ID=2)  # worker_ID=  worker_ID
print(stiffness_fem)
stiffness_fem = stiffness_fem / LA.vector_norm(stiffness_fem)


rew_loss = nn.MSELoss()


def compute_loss(stiffness_fem, stiffness_goal):
    return rew_loss(stiffness_goal, stiffness_fem)


if parameters.unitcell_size == 3 and parameters.stiffness_fem:
      stiffness_goal = torch.tensor([200, 0, 0, 0, 100, 0, 0, 0, 30.7692])  # old one
    #   stiffness_goal = torch.tensor([303.346, 38.075, 3.55271e-15, 38.075, 303.346, 0, 3.64501e-15, 0, 118.711])
elif parameters.unitcell_size == 4 and parameters.stiffness_fem:
      stiffness_goal = torch.tensor(
        [200, -2.46519e-32, 0, 6.16298e-33, 66.6667, 2.22045e-16, 1.44855e-17, 1.22931e-16, 22.2222])
elif parameters.unitcell_size == 5 and parameters.stiffness_fem:
      stiffness_goal = torch.tensor(
        [186.771, 24.6355, -21.7359, 24.6355, 234.741, -24.3362, -21.7359, -24.3362, 82.1964])
stiffness_goal = stiffness_goal / LA.vector_norm(stiffness_goal)



print(compute_loss(stiffness_fem, stiffness_goal))
a.plot(compute_loss(stiffness_fem, stiffness_goal), torch.tensor(42))

stiffness_goal = get_stiffness_goal.stiffness_goal_random_feasible()
print(stiffness_goal)
