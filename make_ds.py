import numpy as np
from tqdm import tqdm
from make_tangle import make_tangle

import pickle


S = 32

import ray
ray.init()

@ray.remote
def make_tangle_remote(i):
   return make_tangle(i)

def make_ds(m, fn):
   data = ray.get([make_tangle_remote.remote(i) for i in range(m)])
   with open(fn, 'wb') as f:
      pickle.dump(data, f)

make_ds(100000, 'train_100000.pkl')
