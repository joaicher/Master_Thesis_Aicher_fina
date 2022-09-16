import gym
import gym.spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.nn.functional
from torch import linalg as LA
import time
import parameters as parameters

if parameters.euler:
    import fem_euler as fem
else:
    import fem as fem

import unitCell as unitCell

rew_loss = nn.MSELoss()


def compute_loss(stiffness_fem, stiffness_goal):
    return rew_loss(stiffness_goal, stiffness_fem)


class Own_Env_v0(gym.Env):
    # possible actions

    # possible positions

    # land on the GOAL position within MAX_STEPS steps
    MAX_STEPS = parameters.max_steps

    # possible rewards
    rew_bad = -2
    rew_step = -1
    rew_final_state = 10

    metadata = {
        "render.modes": ["human"]
    }

    def __init__(self, env_config):
        # the action space ranges [0, 1] where:
        #  `0` move left
        #  `1` move right
        # necessary by gym
        self.done = None
        self.action_space = gym.spaces.Discrete(parameters.number_bars)

        # NB: Ray throws exceptions for any `0` value Discrete
        # observations so we'll make position a 1's based value
        # necessary by gym
        self.observation_space = gym.spaces.MultiBinary(
            parameters.number_bars)  # gym.spaces.Dict({"action_mask": gym.spaces.MultiBinary(parameters.number_bars),
        # "observations": gym.spaces.MultiBinary(parameters.number_bars)})

        # possible positions to chose on `reset()`
        self.unitcell = unitCell.UnitCell(parameters.unitcell_size)
        self.init_positions = torch.ones(parameters.number_bars)
        # self.init_positions.remove(self.goal)

        # NB: change to guarantee the sequence of pseudorandom numbers
        # (e.g., for debugging)
        self.seed()

        self.reset()
        self.worker_ID = str(env_config.worker_index)
        print(self.worker_ID)
        self.overall_steps = 0
        self.infeasible_action = 0
        self.time_old = time.time()

        self.final_state = torch.ones(parameters.number_bars)
        self.final_state[1] = 0
        self.final_state[2] = 0
        self.final_state[3] = 0
        self.final_state[4] = 0

        self.stiffness_fem = torch.zeros(9)
        self.stiffness_goal = torch.zeros(9)
        self.stiffness_goal[0] = 1
        """was checked, result of mesh without inner non-horizontal 3x3 trusses on euler"""
        if parameters.unitcell_size == 3 and parameters.stiffness_fem:
            self.stiffness_goal = torch.tensor([200, 0, 0, 0, 100, 0, 0, 0, 30.7692])  # old one
            # self.stiffness_goal = torch.tensor([303.346, 38.075, 3.55271e-15, 38.075, 303.346, 0, 3.64501e-15, 0, 118.711])
        elif parameters.unitcell_size == 4 and parameters.stiffness_fem:
            self.stiffness_goal = torch.tensor(
                [200, -2.46519e-32, 0, 6.16298e-33, 66.6667, 2.22045e-16, 1.44855e-17, 1.22931e-16, 22.2222])
        elif parameters.unitcell_size == 5 and parameters.stiffness_fem:
            self.stiffness_goal = torch.tensor(
                [186.771, 24.6355, -21.7359, 24.6355, 234.741, -24.3362, -21.7359, -24.3362, 82.1964])
        self.stiffness_goal = self.stiffness_goal / LA.vector_norm(self.stiffness_goal)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.unitcell = unitCell.UnitCell(parameters.unitcell_size)
        self.count = 0
        self.infeasible_action = 0

        # for this environment, state is simply the position
        self.state = self.unitcell.transform_to_bars(parameters.unitcell_size)
        self.reward = torch.tensor(0)
        self.done = False
        self.info = {}

        return self.state

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete

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
            self.count += 1
            self.overall_steps += 1

            # reward function: reward is rew_loss if rew_loss of next state is less than rew_loss of current state, else done=True
            """define distance of current state"""
            current_step_fem, _ = fem.compute_rFEM(100, steps_counter=self.overall_steps, worker_ID=self.worker_ID)
            current_step_fem = current_step_fem/LA.vector_norm(current_step_fem)
            curr_rew_loss = compute_loss(current_step_fem, self.stiffness_goal)

            """perform action"""
            try:
                self.unitcell.bar_removed(action, parameters.unitcell_size)
                self.state = self.unitcell.transform_to_bars(parameters.unitcell_size)

                if parameters.euler:
                    self.unitcell.save('mesh' + self.worker_ID)
                else:
                    self.unitcell.save(parameters.path_laptop + '/beamHomogenization/mesh' + self.worker_ID)

                """compute stiffness"""
                self.stiffness_fem, _ = fem.compute_rFEM(100, steps_counter=self.overall_steps, worker_ID=self.worker_ID)
                self.stiffness_fem = self.stiffness_fem / LA.vector_norm(self.stiffness_fem)
            except:
                self.reward = torch.tensor(-2)
                self.infeasible_action += 1
                self.info["infeasible_action"] = self.infeasible_action

            #reward definition: reward is rew_loss if rew_loss of next state is less than rew_loss of current state, else done=True
            if curr_rew_loss > compute_loss(self.stiffness_fem, self.stiffness_goal) and self.count > 4:
                self.reward = - compute_loss(self.stiffness_fem, self.stiffness_goal)
                self.done = True
            else:
                self.reward = - compute_loss(self.stiffness_fem, self.stiffness_goal)

            self.info["dist"] = (compute_loss(self.stiffness_fem, self.stiffness_goal)).item()
            self.info["overallsteps"] = self.overall_steps

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        # NaN checks:
        if self.reward != self.reward:
            self.reward = torch.tensor(-3)
            print("reward NaN error, action: & step", action, self.count)

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



