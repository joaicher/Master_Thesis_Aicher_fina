import parameters
if parameters.euler:
    import fem_euler as fem
else:
    import fem as fem
import random
import unitCell
import torch
from torch import linalg as LA
import time


def stiffness_goal_random_feasible(worker_ID):
    """
    Returns a random stiffness goal that is feasible for the given unit cell size.
    :return: torch.tensor(9) with stiffness goal
    """
    #chose a random number of bars to be removed; boundaries might be adapted
    num_bar_removed = random.randint(0, parameters.number_bars - parameters.bars_remaining)

    a = unitCell.UnitCell(parameters.unitcell_size)
    # remove random bars
    for i in range(num_bar_removed):
        bar_rem = random.randint(0, parameters.number_bars - 1)
        try: # if the bar is already removed, the program will crash
            a.bar_removed(bar_rem, parameters.unitcell_size)
        except:
            pass
    # plot the unit cell if required
    if parameters.check:
        a.plot(torch.tensor(num_bar_removed), torch.tensor(42))

    # save the mesh file in the same directory where the BeamHomogenization-fromFile2D is located
    if parameters.euler:
        a.save('mesh' + worker_ID)
    else:
        a.save('/Users/Johannes/Library/CloudStorage/OneDrive-Persönlich/Dokumente/ETH-Studium-Gesamt/MasterThesis/ae108'
               '-legacy/build/drivers/beamHomogenization/mesh1')
    # compute the stiffness matrix
    if not parameters.check:
        time.sleep(0.05)
        stiffness_goal, _ = fem.compute_rFEM(2, 1, worker_ID=worker_ID)
        stiffness_goal = stiffness_goal /LA.vector_norm(stiffness_goal)
    else:
        # compute random numbers
        stiffness_goal = torch.rand(9)
        stiffness_goal[1] = stiffness_goal[3]
        stiffness_goal[2] = stiffness_goal[6]
        stiffness_goal[5] = stiffness_goal[7]
        print("random numbers as stiffness goal")

    return stiffness_goal




