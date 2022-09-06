Implementation of a Reinforcement Learning Algorithm to find a truss structure matching a target stiffness .
Framework: ray, rllib library;

[]: # Language: python

What files are relevant & up to date:
- train.py: trains the model
- parameters.py: contains the parameters of the model that are frequently changed (e.g. number of iterations, truss size etc.)
- env_class (_2/_3/env_03, env_goal): 
  - the environment for algorithm; 
  - most of the relevant code is in those files, i.e. the step, state, reward definition, termination criterion etc.
- parameters.env_class = 
    # 0: env_goal - changing stiffness target, threshold final state;
    # 1: threshold approach;
    # 2: derivative approach;
    # 3: fixed goal but final state chosen by network - not 3 && custom model!!;
    # 4: mix from 0 and 3: changing stiffness target with final state chosen by network
- fem.py: runs the BeamHomogenization-fromFile2D on the local laptop, in Docker container; pathes needs to be adapted for different computer
- fem_euler.py: runs the BeamHomogenization-fromFile2D on euler, where the BeamHomogenization-fromFile2D is in the same directory as the other .py files
- get_stiffness_goal.py: creates a feasible, random stiffness goal (for env_goal, env_03)
- custom_model.py: creates a custom model for the network, is used for action space masking
- unitCell.py: the unit cell class, contains many functions for different state representations etc. 

for the comments in the enviroments, env_03 is the best commented, since it is the most "advanced" environment
with changing stiffness goal and termination chosen by network;


Folder Structure on euler:
To run the code on euler, the ae-108 BeamHomogenization-fromFile2D must be compiled.
This repo (all of the python files) should be placed in the build/drivers/beamHomogenization folder, where the BeamHomogenization-fromFile2D is located.
Pathes must be adapted in the fem_euler.py file.
If in doubt, use ctrl+f to search for "jaicher" in the code and replace it with your username.
The tensorboard output will be written to home/username/ray_results/



the other files are not that relevant and sometimes not up to date; be especially careful when using them. 













# gym_example

Example implementation of an [OpenAI Gym](http://gym.openai.com/) environment,
to illustrate problem representation for [RLlib](https://rllib.io/) use cases.

## Usage

Clone the repo and connect into its top level directory.

To initialize and run the `gym` example:

```
pip install -r requirements.txt
pip install -e gym-example

python sample.py
```

To run Ray RLlib to train a policy based on this environment:

```
python train.py
```


## Kudos

h/t:

  - <https://github.com/DerwenAI/gym_trivial>
  - <https://github.com/DerwenAI/gym_projectile>
  - <https://github.com/apoddar573/Tic-Tac-Toe-Gym_Environment/>
  - <https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa>
  - <https://github.com/openai/gym/blob/master/docs/creating-environments.md>
  - <https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai>
