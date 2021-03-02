import os
import json
import random
import numpy as np
import torch


def save_to_json(filename, data):
    with open(filename, 'w') as f:
        return json.dump(data, f)


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def freeze_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.deterministic = deterministic
    torch.cuda.benchmark = benchmark


def set_max_threads(max_threads: int = 32) -> None:
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
