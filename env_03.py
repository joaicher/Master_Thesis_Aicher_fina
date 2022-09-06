import gym
import gym.spaces
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch import linalg as LA
import time
import parameters as parameters
import get_stiffness_goal

if parameters.euler:
    import fem_euler as fem
else:
    import fem as fem

import unitCell as unitCell

# how the distance between stiffness target and state's stiffness is computed
# currently not complicated, but can be changed to be more complex
rew_loss = nn.MSELoss()


def compute_loss(stiffness_fem, stiffness_goal):
    return rew_loss(stiffness_goal, stiffness_fem)


class Own_Env_v0(gym.Env):
    # land on the GOAL position within MAX_STEPS steps
    MAX_STEPS = parameters.max_steps

    metadata = {
        "render.modes": ["human"]
    }

    def __init__(self, env_config):

        # necessary by gym
        self.done = None
        self.action_space = gym.spaces.Discrete(parameters.number_bars + 1)

        # necessary by gym ##
        # observation space is a dict, multi binary whether bar exists or not, and box with stiffness goal
        self.observation_space = gym.spaces.Dict(
            {
                'bars': gym.spaces.MultiBinary(parameters.number_bars),
                'stiffness_goal': gym.spaces.Box(high=1.01, low=-1.01, shape=(9,), dtype=np.float32),
            }
        )

        # possible positions to chose on `reset()`
        self.unitcell = unitCell.UnitCell(parameters.unitcell_size)
        self.init_positions = torch.ones(parameters.number_bars)
        # self.init_positions.remove(self.goal)

        # NB: change to guarantee the sequence of pseudorandom numbers
        # (e.g., for debugging)
        self.seed(42)

        self.worker_ID = str(env_config.worker_index)
        print(self.worker_ID)

        self.overall_steps = 0
        self.infeasible_action = 0
        self.final_action = 0
        self.time_old = time.time()

        # find initial stiffness
        self.stiffness_fem = torch.zeros(9)
        if parameters.stiffness_fem:
            if parameters.euler:
                self.unitcell.save('mesh' + self.worker_ID)
            else:
                self.unitcell.save('/Users/Johannes/Library/CloudStorage/OneDrive-Persönlich/Dokumente/ETH'
                                   '-Studium-Gesamt/MasterThesis/ae108-legacy/build/drivers/beamHomogenization/mesh' + self.worker_ID)
            self.stiffness_fem, _ = fem.compute_rFEM(100, steps_counter=self.overall_steps,
                                                     worker_ID=self.worker_ID)
            self.stiffness_fem = self.stiffness_fem / LA.vector_norm(self.stiffness_fem)

        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns the state, sets some counters to zero, and resets the unitcell to be fully connected
        -------
        observation (object): the initial observation of the space.
        """
        self.count = 0
        #self.infeasible_action = 0
        # define fully connected unitcell as initial state
        self.unitcell = unitCell.UnitCell(parameters.unitcell_size)

        # reset stiffness goal
        try:
            self.stiffness_goal = get_stiffness_goal.stiffness_goal_random_feasible(worker_ID=self.worker_ID)
            if parameters.check:
                print(self.stiffness_goal)
        except:
            print("Feasible stiffness goal failed")
            self.stiffness_goal = torch.rand(9)
            self.stiffness_goal[1] = self.stiffness_goal[3]
            self.stiffness_goal[2] = self.stiffness_goal[6]
            self.stiffness_goal[5] = self.stiffness_goal[7]

        # for this environment, state is the position and the stiffness goal
        self.state = {'bars': self.unitcell.transform_to_bars(parameters.unitcell_size).numpy(),
                      'stiffness_goal': self.stiffness_goal.numpy(),
                      }

        self.reward = torch.tensor(0)
        self.final_action = 0
        self.done = False
        self.info = {}
        if parameters.check:
            print(self.state)

        return self.state

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete, chosing which bar to remove

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.done:
            # code should never reach this point
            print("EPISODE DONE!!!")

        elif self.count == self.MAX_STEPS:
            self.done = True

        else:
            assert self.action_space.contains(action)

            # type check, that action is int
            if not isinstance(action, (int, np.integer, torch.int32)):
                print("action is not int", type(action), action)
                action = round(action)
            self.count += 1
            self.overall_steps += 1

            # compute state for first step
            if self.count == 1 and parameters.stiffness_fem:
                try:
                    self.stiffness_fem, _ = fem.compute_rFEM(100, steps_counter=self.overall_steps,
                                                             worker_ID=self.worker_ID)
                except:
                    print("FEM failed in env_class, step 0")
                    self.stiffness_fem, _ = fem.compute_rFEM(100, steps_counter=self.overall_steps,
                                                             worker_ID=self.worker_ID)
                self.stiffness_fem = self.stiffness_fem / LA.vector_norm(self.stiffness_fem)

            """
            define final state
            when to stop is (for this environment) chosen by the network as additional action
            the network must choose to stop twice (getting the "final reward" twice)
            by that, it is enforced that the final state is as close as possible to the goal
            """
            if self.final_action == 1 and action == (parameters.number_bars):
                self.reward = (1 - parameters.loss_param * compute_loss(self.stiffness_fem, self.stiffness_goal))
                self.unitcell.plotsave("final state", self.reward,
                                       compute_loss(self.stiffness_fem, self.stiffness_goal))
                self.done = True
            elif action == parameters.number_bars:
                self.final_action += 1
                self.reward = (1 - parameters.loss_param * compute_loss(self.stiffness_fem, self.stiffness_goal))

            else:
                """perform action"""
                try:
                    # remove bar
                    self.unitcell.bar_removed(action, parameters.unitcell_size)
                    self.state['bars'] = self.unitcell.transform_to_bars(parameters.unitcell_size).numpy()

                    # save mesh
                    if parameters.euler:
                        self.unitcell.save('mesh' + self.worker_ID)
                        time.sleep(0.1)
                    else:
                        self.unitcell.save('/Users/Johannes/Library/CloudStorage/OneDrive-Persönlich/Dokumente/ETH'
                                           '-Studium-Gesamt/MasterThesis/ae108-legacy/build/drivers/beamHomogenization/mesh' + self.worker_ID)

                    """ compute reward """
                    if parameters.stiffness_fem:
                        start_time_fem = time.time()
                        try:
                            self.stiffness_fem, _ = fem.compute_rFEM(100, steps_counter=self.overall_steps,
                                                                     worker_ID=self.worker_ID)
                        except:
                            print("FEM failed in env_class")
                            time.sleep(0.2)
                            self.stiffness_fem, _ = fem.compute_rFEM(100, steps_counter=self.overall_steps,
                                                                     worker_ID=self.worker_ID)
                        end_time_fem = time.time()
                        if parameters.time_measure:
                            print("overall fem time", end_time_fem - start_time_fem)
                        self.stiffness_fem = self.stiffness_fem / LA.vector_norm(self.stiffness_fem)
                    else:
                        self.stiffness_fem[0] = self.unitcell.hor_count() / self.unitcell.count_edges()

                        # reward is positive since the agent should do as many steps as necessary, the high reward of
                        # the final state drives it to stop as soon as possible, the reward passed at action = "hold"
                        # is higher than the reward passed for a sub-optimal solution
                        self.reward = (
                                1 - parameters.loss_param * compute_loss(self.stiffness_fem, self.stiffness_goal))

                    # define final state here to avoid duplicated code for the "horizontal bars problem"
                    if self.stiffness_fem[0] == 1 and not parameters.stiffness_fem:
                        self.reward = torch.tensor(20)
                        self.done = True

                    self.info["dist"] = ((self.stiffness_fem - self.stiffness_goal).norm()).item()
                    self.info["overallsteps"] = self.overall_steps

                # that happens especially if the action is infeasible (bar already removed)
                except:
                    self.reward = torch.tensor(-5)
                    self.infeasible_action += 1
                    self.info["infeasible_action"] = self.infeasible_action

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)
            self.done = True

        # NaN checks:
        if self.reward != self.reward:
            self.reward = torch.tensor(-3)
            print("reward NaN error, action: & step", action, self.count)

        # print some intermeidate information
        if self.overall_steps % 10000 == 0:
            print("overall steps performed and number of infeasible actions: ", self.overall_steps,
                  self.infeasible_action)
            print("time for 100000 actions", time.time() - self.time_old)
            self.time_old = time.time()

        return [self.state, self.reward.item(), self.done, self.info]

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        s = "reward: {}  info: {}"
        print(s.format(self.reward.item(), self.info))
        print(self.state)
        print(self.overall_steps)
        # plot unitcell
        self.unitcell.plotsave("sample_structure", self.reward, self.stiffness_fem)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass
