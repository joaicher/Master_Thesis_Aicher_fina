#!/usr/bin/env python
# encoding: utf-8

import gym
import os
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env
import shutil
import time
import parameters
import custom_model
from ray.rllib.models import ModelCatalog
from gym_env_pers.envs.example_env import Example_v0

ModelCatalog.register_custom_model("custom_torch_model", custom_model.CustomTorchModel)



# main runs the forward run through the pre-trained model
def  main():
    print("Custom Model:", parameters.custom_mod_binary)
    print("Number workers:", parameters.num_workers, " Unitcell size: ", parameters.unitcell_size)

    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))

    # start Ray -- add `local_mode=True` here for debugging
    context = ray.init(ignore_reinit_error=True)
    print(f"Dashboard url: http://{context.address_info['webui_url']}")

    # register the custom environment
    select_env = "example-v0"
    register_env(select_env, lambda config: Example_v0())
    register_env("own_env-v0", env_creator)

    # configure the environment and create agent

    if parameters.custom_mod_binary:
        config = {"framework": "torch",
                  "model": {
                      "custom_model": "custom_torch_model",
                      "fcnet_hiddens": [256, 256, parameters.number_bars],
                      "no_final_linear": True,
                      # Extra kwargs to be passed to your model's c'tor.
                      "custom_model_config": {},
                  },
                  "num_workers": parameters.num_workers,
                  "ignore_worker_failures": True,
                  "seed": 42,
                  }
    else:
        config = dqn.DEFAULT_CONFIG.copy()  # see also https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
        print(config)
        config["ignore_worker_failures"] = True
        config["num_workers"] = parameters.num_workers
        # each process gets multiple envs and batches policy gradients between them
        config["num_envs_per_worker"] = 1
        config["log_level"] = "WARN"
        config["seed"] = 42



    agent = dqn.DQNTrainer(config=config, env="own_env-v0")  # ppo.PPOTrainer(config, env=select_env) #

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    # show the model's structure
    print(model.summary())

    # apply the trained policy in a rollout
    chkpt_file = "tmp/exa/checkpoint_000010/checkpoint-" + parameters.n_interations
    agent.restore(chkpt_file)
    """raised error!!!!"""
    env = gym.make('example-v0')

    state = env.reset()
    sum_reward = 0
    n_step = 20


    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0