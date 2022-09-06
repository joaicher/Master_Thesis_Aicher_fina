#!/usr/bin/env python
# encoding: utf-8

"""this file should tune the hyperparameters of the train.py file"""

import gym
import os
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import shutil
import time
import parameters

if parameters.env_class == 0:
    import env_goal as env_class
if parameters.env_class == 1:
    import env_class
elif parameters.env_class == 2:
    import env_class_2 as env_class
elif parameters.env_class == 3:
    import env_class_3 as env_class

import custom_model
from ray.rllib.models import ModelCatalog


def main():
    print("Custom Model:", parameters.custom_mod_binary)
    print("Number workers:", parameters.num_workers, " Unitcell size: ", parameters.unitcell_size)
    print("Environment:", parameters.env_class)

    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))

    # register the custom environment
    register_env("own_env-v0", env_creator)

    # start Ray -- add `local_mode=True` here for debugging
    if parameters.euler:
        context = ray.init(ignore_reinit_error=True) #memory=72 * 1024 * 1024 * 1024
    else:
        context = ray.init(ignore_reinit_error=True, local_mode=True)

    # define the parameters to be searched
    search_space = {
        # the network size
        "fcnet_hiddens": [tune.choice([64, 128, 256]), tune.choice([64, 128, 256])],
    }

    # configure the environment and create agent
    config = dqn.DEFAULT_CONFIG.copy()  # see also https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
    print(config)
    config["ignore_worker_failures"] = True
    config["num_workers"] = parameters.num_workers
    # each process gets multiple envs and batches policy gradients between them
    config["num_envs_per_worker"] = 1
    config["log_level"] = "WARN"
    config["seed"] = 42
    config["model"] = {
        "fcnet_hiddens": search_space["fcnet_hiddens"],
    }
    config["env"] = "own_env-v0"

    #agent = dqn.DQNTrainer(config=config, env="own_env-v0")

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = parameters.n_interations

    # use tune.run to train the agent and the corresponding hyperparameters
    analysis = tune.run("DQN", scheduler=ASHAScheduler(metric="episode_reward_mean", mode="max"), config=config,
                        stop={"training_iteration": n_iter})

    print("best hyperparameters: ", analysis.best_config)


def env_creator(env_config):
    return env_class.Own_Env_v0(env_config)


if __name__ == "__main__":
    main()
