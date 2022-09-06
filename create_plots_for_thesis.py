"""
this file should create the plots for the thesis
one big part will be the investigation of the performance of the pretrained model for many stiffness inputs
"""

import gym
import os
import ray
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
import torch
from ray.tune.registry import register_env
import shutil
import time
import parameters
import env_con
import math
import matplotlib.pyplot as plt


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


def env_creator(env_config):
    return env_class.Own_Env_v0(env_config)


def main():
    register_env("own_env-v0", env_creator)
    ray.init(ignore_reinit_error=True, local_mode=True)

    config = dqn.DEFAULT_CONFIG.copy()
    config["ignore_worker_failures"] = True
    config["num_workers"] = 1
    # each process gets multiple envs and batches policy gradients between them
    config["num_envs_per_worker"] = 1

    agent = dqn.DQNTrainer(config=config, env="own_env-v0")

    # apply the trained policy in a rollout
    # load the checkpoint
    chkpt_file = "checkpoint_500/checkpoint-500"
    if parameters.euler:
        chkpt_root = "tmp/exa"
        n = parameters.n_interations
        #agent.restore(chkpt_root + "/checkpoint_" + str(n) + "/checkpoint-" + str(n))
        agent.restore(chkpt_root + "/checkpoint_" + str(500) + "/checkpoint-" + str(500))
    else:
        agent.restore(chkpt_file)

    # reset the environment to initial state
    env = env_class.Own_Env_v0(env_config=env_con)
    state = env.reset()
    sum_reward = 0
    n_step = parameters.max_steps

    # apply the policy in a rollout
    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward


        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            env.render()
            state = env.reset()
            sum_reward = 0

    # lets evalutate the performance on random feasible goals; only works for env0 and env4
    if parameters.env_class == 0 or parameters.env_class == 4:
        n_eval = 100
        sum_reward = 0
        reward_list = []
        dist_list = []
        MSE_list = []
        # lets compute the mean reward for each episode (starting with env.reset)
        for i in range(n_eval):
            state = env.reset()
            for step in range(n_step):
                action = agent.compute_action(state)
                state, reward, done, info = env.step(action)
                sum_reward += reward


                if done == 1:
                    # report at the end of each episode
                    reward_list.append(sum_reward)
                    dist_list.append(info["dist"])
                    MSE_list.append(info["MSE_loss"])
                    sum_reward = 0
                    break

            reward_list.append(sum_reward)
            dist_list.append(info["dist"])
            MSE_list.append(info["MSE_loss"])
            sum_reward = 0

        # compute the avg. performance and variance
        avg_reward = sum(reward_list) / n_eval
        var_reward = sum((x - avg_reward)**2 for x in reward_list) / n_eval
        print("avg reward", avg_reward)
        print("var reward", var_reward)

        # compute the avg. distance and variance
        avg_dist = sum(dist_list) / n_eval
        var_dist = sum((x - avg_dist)**2 for x in dist_list) / n_eval
        print("avg dist", avg_dist)
        print("var dist", var_dist)

        # compute the avg. MSE and variance
        avg_MSE = sum(MSE_list) / n_eval
        var_MSE = sum((x - avg_MSE)**2 for x in MSE_list) / n_eval
        print("avg MSE", avg_MSE)

        # save dist and reward to csv file
        with open("dist_reward.csv", "w") as f:
            for i in range(n_eval):
                f.write(str(dist_list[i]) + "," + str(reward_list[i]) + "\n")
        with open("MSE.csv", "w") as f:
            for i in range(n_eval):
                f.write(str(MSE_list[i]) + "\n")
        with open("MSE_sqrt.csv", "w") as f:
            for i in range(n_eval):
                f.write(str(torch.sqrt(torch.tensor(MSE_list[i]))) + "\n")

        # plot the dist and reward
        plt.plot(dist_list, reward_list, ".")
        plt.savefig("dist_reward.png")









if __name__ == "__main__":
    main()
