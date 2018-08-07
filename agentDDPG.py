from nnDDPG import *

class DDPGAgent:
    def __init__(self, config):
        DDPGAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)