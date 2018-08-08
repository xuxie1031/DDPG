import torch

from task import *
from replay import *
from randomProcess import *
from nnUtils import *
from nnDDPG import DDPGNet
from config import Config

from agentDDPG import DDPGAgent


def run_ddpg():
    config = Config()
    config.task_fn = lambda name: Roboschool(name)
    config.network_fn = lambda state_dim, action_dim: DDPGNet(
        state_dim, action_dim,
        actor_body = FCBody(state_dim, hidden_units=(300, 200), gate=torch.tanh),
        critic_body = FCBodyWithAction(state_dim, action_dim, hidden_state_dim=400, hidden_units=(300, ), gate=torch.tanh),
        actor_opt_fn = lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn = lambda params: torch.optim.Adam(params, lr=1e-3),
	gpu = 0
    )
    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = .99
    config.random_process_fn = lambda action_dim: OrnsteinUhlenbeckProcess(size=(action_dim, ), std=LinearSchedule(.2))
    config.min_replay_size = 64
    config.target_network_mix = 1e-3

    config.episodes_num = 10000
    agent = DDPGAgent(config)
    agent.run_agent()


if __name__ == '__main__':
    run_ddpg()
