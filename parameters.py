from ray import tune
import torch

# the number of bars in the fully connected lattice
number_bars = 33  # 33 for 4x4, 16 for 3x3, 56 for 5x5; its n = (N-1)*(3N-1)
# the size of the quadratic unitcell
unitcell_size = 4
# whether to use the BeamHomogenization-fromFile2D to compute the stiffness
# (if False, the goal is to remove all vertical bars)
stiffness_fem = True
# whether to run the BeamHomogenization-fromFile2D on Euler or on your laptop.
# if laptop, the paths need to match (relevant in environment file (i.e. env_class_3) and fem.py/fem_euler.py)
euler = True
# time measure = True will print a lot of runtime measurements in the console
time_measure = False
#the max number of steps per episode
max_steps = 20 #12 for 3x3, 15 for stiffness_fem=False and 3x3
# the number of workers running in parallel (should be #cores-1)
if not euler:
    num_workers = 2
else:
    num_workers = 14
# whether to use action masking (custom_mod_binary = True) or not
custom_mod_binary = False
# number of iterations; number of episodes per iteration is chosen by ray
n_interations = 500

# 0: env_goal - changing stiffness target, threshold final state;
# 1: threshold approach;
# 2: derivative approach;
# 3: fixed goal but termination chosen by network - not 3 && custom model!!;
# 4: mix from 0 and 3: changing stiffness target with final state chosen by network
# 4  do not use 4 with custom model!!;
env_class = 3
if number_bars == 16:
    goal_threshold = 0.0001 # 1% since sqrt
else:
    goal_threshold = 0.0001 #3x3: 0.0000001
# check = False disables FEM computation in get_stiffness_goal.py and prints more results
check = False
# loss param is used when the network can choose the stop action:
# reward = 1 - loss_param*(distance betw. goal and state)
loss_param = 5
# irrelevant parameter as for now
conv_model = False

# minimum number of remaining bars if target is chosen randomly;
bars_remaining = 7

# if an infeasible, fixed goal should be taken, for env3:
infeasible_fixed_goal_bin = False
infeasible_fixed_goal = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0])

path_euler = "/cluster/home/jaicher/MA/ae108-legacy/build/drivers/"

# this is the path to the beamHomogenization directory, that should consist the BeamHomogenization-fromFile2D executable
path_laptop = '/Users/Johannes/Library/CloudStorage/OneDrive-Persönlich/Dokumente/ETH-Studium-Gesamt/MasterThesis' \
              '/ae108-legacy/build/drivers/ '

