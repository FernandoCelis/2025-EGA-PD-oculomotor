import numpy as np
import random
import datetime
import torch


def get_date():
    now = datetime.datetime.now()
    return now.strftime("%y%m%d-%H%M%S")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
