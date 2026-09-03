"""
This script shows how to train the Unitree Go2 velocity tracking task with PPO in MuJoCo Warp.

"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.environments.mujoco_warp_envs import Go2Walk
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import (
    FeedForwardNetwork,
    ActorNetwork,
)
from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor
from mushroom_rl.utils.torch_utils import TorchUtils


def experiment(
    n_epochs,
    n_steps,
    n_steps_per_fit,
    n_episodes_test,
    n_envs,
    use_graph_capture=True,
    seed=None,
):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    assert torch.cuda.is_available(), "MuJoCo Warp requires a CUDA device."
    assert n_envs >= 2, "n_envs must be at least 2."

    TorchUtils.set_default_device("cuda:0")

    # MDP
    mdp = Go2Walk(num_envs=n_envs, use_graph_capture=use_graph_capture)

    logger = Logger(f"{PPO.name()}_{mdp.name()}", results_dir=None, seed=seed)
    logger.log_experiment_info(
        PPO,
        mdp,
        n_epochs=n_epochs,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=n_episodes_test,
        n_envs=n_envs,
    )

    # Settings
    actor_lr = 3e-4
    critic_lr = 3e-4
    n_features = 128
    batch_size = 1024
    n_epochs_policy = 5
    eps = 0.2
    lam = 0.95
    std_0 = 1.0
    ent_coeff = 0.001

    # Policy
    policy = GaussianTorchPolicy(
        ActorNetwork,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=std_0,
        n_features=n_features,
        gain_scale=0.1,
    )

    # Agent
    critic_params = dict(
        network=FeedForwardNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        gain_scale=0.1,
        batch_size=batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    agent = PPO(
        mdp.info,
        policy,
        critic_params=critic_params,
        actor_optimizer={"class": optim.Adam, "params": {"lr": actor_lr}},
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps,
        lam=lam,
        ent_coeff=ent_coeff,
    )

    agent.add_core_preprocessor(StandardizationPreprocessor(mdp.info))

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    E = agent.policy.entropy().item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        E = agent.policy.entropy().item()

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-envs", type=int, default=100, help="number of parallel environments"
    )
    parser.add_argument(
        "--fragment-length",
        type=int,
        default=1000,
        help="on-policy steps collected per environment per fit. "
        "n_steps_per_fit is derived from this so that "
        "changing --n-envs does not silently shorten the "
        "GAE horizon",
    )
    parser.add_argument(
        "--fits-per-epoch", type=int, default=10, help="policy updates per epoch"
    )
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument(
        "--no-graph-capture",
        action="store_false",
        dest="graph_capture",
        help="disable CUDA graph capture",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the experiment, random when not given",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n_steps_per_fit = args.n_envs * args.fragment_length
    n_steps = n_steps_per_fit * args.fits_per_epoch

    experiment(
        n_epochs=args.n_epochs,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=10,
        n_envs=args.n_envs,
        use_graph_capture=args.graph_capture,
        seed=args.seed,
    )
