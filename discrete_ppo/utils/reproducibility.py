import torch
import numpy as np


def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
