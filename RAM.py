import ray
import os
from ray.rllib.agents.ppo import PPOTrainer
import ray.rllib.agents.ppo as ppo
ray.shutdown()
info = ray.init(ignore_reinit_error=True, dashboard_port=8266)
print(info.address_info['webui_url'])
#print('Dashboard URL: http://{}'.format(ray.get_webui_url()))
print("Dashboard URL: http://{}".format(info.address_info["webui_url"]))
import shutil

CHECKPOINT_ROOT = "tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

SELECT_ENV = "Taxi-v3"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = ppo.PPOTrainer(config, env=SELECT_ENV)

N_ITER = 3
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
  result = agent.train()
  file_name = agent.save(CHECKPOINT_ROOT)

  print(s.format(
    n + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))

policy = agent.get_policy()
model = policy.model
print(model.base_model.summary())