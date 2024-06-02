import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from make_tangle import make_tangle
from ema_pytorch import EMA
from einops import einsum
import matplotlib.pyplot as plt

import pickle

torch.autograd.set_detect_anomaly(True)

S = 32

import ray
ray.init(num_cpus=6)

@ray.remote
def make_tangle_remote(i):
   return make_tangle(i)

def make_ds(m, fn):
   data = ray.get([make_tangle_remote.remote(i) for i in range(m)])
   with open(fn, 'wb') as f:
      pickle.dump(data, f)

make_ds(100000, 'train_100000.pkl')
