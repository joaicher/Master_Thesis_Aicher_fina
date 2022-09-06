from gym.spaces import Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch

import parameters

torch, nn = try_import_torch()

'''inspired by: 
https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py 
https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505  
https://docs.ray.io/en/latest/rllib/rllib-models.html#'''

"""the pytorch standard dqn model: 
https://github.com/ray-project/ray/blob/0c469e490e0ed5e6ca848c627f3b852382e2bf2a/rllib/algorithms/dqn/dqn_torch_model.py

and the corresponding ModelV2:
https://github.com/ray-project/ray/blob/0c469e490e0ed5e6ca848c627f3b852382e2bf2a/rllib/models/modelv2.py"""


class CustomTorchModel(TorchModelV2, nn.Module):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        # assert (
        #     isinstance(orig_space, Dict)
        #     and "action_mask" in orig_space.spaces
        #     and "observations" in orig_space.spaces
        # )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]})



        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state
        logits = logits * action_mask
        # Convert action_mask into a [0.0 || -inf]-type mask.
        #inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        #inf_mask = inf_mask.unsqueeze(1)
        #masked_logits = logits  + inf_mask
        # Return masked logits.
        return logits, state

    def value_function(self):
        return self.internal_model.value_function()


class Own_model(nn.Module):
    def __init__(self):
        super(Own_model, self).__init__()
        self.fc1 = nn.Linear(parameters.number_bars, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, parameters.number_bars)

    def forward(self, input_dict, state, seq_lens):
        x = state
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act1(x)
        x = self.fc3(x)
        return x
