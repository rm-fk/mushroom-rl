import torch
import warp as wp
from pathlib import Path

from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.core.spaces import Box

class Go2Base(MuJoCoWarp):
    def __init__(
            self,
            num_envs,
            gamma,
            horizon, 
            use_graph_capture,
            ):
        super().__init__(
                num_envs=num_envs,
                gamma=gamma,
                horizon=horizon,
                use_graph_capture=use_graph_capture.)
    def _preprocess_action():
        pass

    def _is_healthy():
        pass



