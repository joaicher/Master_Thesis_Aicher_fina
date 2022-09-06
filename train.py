#!/usr/bin/env python
# encoding: utf-8
"""
main script for training the ray RL model
created by Johannes Aicher
inspired by https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5
however, the repos file layout was changed: the environment file is now in the same directory as the main script;
that increases the efficiency of the workflow when using euler, but rollout doesnt work anymore
that can be changed easily, the file structure is still in the repo; pls check that the code from env_class.py is the same as
in gym-example.gym_env_pers.envs.example_env.py
"""

import gym
import os
import ray
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import shutil
import time
import parameters

if parameters.env_class == 0:
    import env_goal as env_class
elif parameters.env_class == 1:
    import env_class
elif parameters.env_class == 2:
    import env_class_2 as env_class
elif parameters.env_class == 3:
    import env_class_3 as env_class
elif parameters.env_class == 4:
    import env_03 as env_class

# custom model includes action space masking,
# but doesnt proof to be superior to the default model
import custom_model
from ray.rllib.models import ModelCatalog

if not parameters.euler:
    from gym_env_pers.envs.example_env import Example_v0

ModelCatalog.register_custom_model("custom_torch_model", custom_model.CustomTorchModel)


def main():
    print("Custom Model:", parameters.custom_mod_binary)
    print("Number workers:", parameters.num_workers, " Unitcell size: ", parameters.unitcell_size)
    print("Environment:", parameters.env_class)

    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    # shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    if parameters.euler:
        context = ray.init(ignore_reinit_error=True)
    else:
        context = ray.init(ignore_reinit_error=True, local_mode=True)
    # print(f"Dashboard url: http://{context.address_info['webui_url']}")

    # register the custom environment
    # select_env = "example-v0"
    # register_env(select_env, lambda config: Example_v0())
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
        # fully connected network for 1D input space, convolutional network for 2D input space
        # see also https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
        config = dqn.DEFAULT_CONFIG.copy()
        print(config)
        config["ignore_worker_failures"] = True
        config["num_workers"] = parameters.num_workers
        # each process gets multiple envs and batches policy gradients between them
        config["num_envs_per_worker"] = 1
        config["log_level"] = "WARN"
        config["seed"] = 42

        # for trying different model sizes, description in my thesis
        # config["model"] = {
        #     "fcnet_hiddens": [512, 512, 512],
        #     "fcnet_activation": "relu",
        # }
        # config["dueling"] = False
        # config["hiddens"] = []

    # setup the agent
    agent = dqn.DQNTrainer(config=config, env="own_env-v0")  # ppo.PPOTrainer(config, env=select_env) #

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    # n_iter gives number of iterations; #episodes per iteration is not constant
    n_iter = parameters.n_interations

    # train a policy with RLlib
    start_time_train = time.time()
    restore_counter = 0
    for n in range(n_iter):
        time_iter = time.time()
        try:
            result = agent.train()
        except AssertionError:
            restore_counter += 1
            # check if checkpoint exists
            if os.path.exists(chkpt_root):
                print("Checkpoint exists, iteration:", n)
                # load checkpoint in directory chkpt_root plus number of checkpoint
                # reset agent
                agent = dqn.DQNTrainer(config=config, env="own_env-v0")
                print("Agent restored at interation:", n)
                agent.restore(chkpt_root + "/checkpoint_" + str(n) + "/checkpoint-" + str(n))
                print("Loaded checkpoint")
                result = agent.train()
                print("AssertionError")
            else:
                print("AssertionError, no checkpoint")
                continue
        except:
            print("Error in training iteration", n)
            continue
        print("iteration time: ", time.time() - time_iter)
        # remove the old checkpoint files
        if n > 1:
            try:
                os.remove(chkpt_root + "/checkpoint_" + str(n-1) + "/checkpoint-" + str(n-1))
            except:
                print("Error in removing old checkpoint")
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
        ))

    end_time_train = time.time()
    print("seconds for this run:", end_time_train - start_time_train)
    print(list(result))


def env_creator(env_config):
    return env_class.Own_Env_v0(env_config)


if __name__ == "__main__":
    main()

'''
    # config["model"] = {
    #     "fcnet_hiddens": [64, 64],
    #     "fcnet_activation": "relu",
    # }
    '''
