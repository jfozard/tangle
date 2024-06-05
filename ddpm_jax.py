import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import wandb
import time
import tensorflow as tf

import pickle
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
#from torchvision import datasets, transforms
#from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from make_tangle import make_tangle
#from ema_pytorch import EMA
import matplotlib.pyplot as plt
import pickle
import tqdm
import utils
import functools
from flax import jax_utils
from flax.training import common_utils

from scipy.optimize import linear_sum_assignment
from imageio import imwrite

import matplotlib.pyplot as plt

import ray
ray.init()#num_cpus=6)




def match_paths(x, x0):
    print('match', x.shape, x0.shape)
    cost_mask = batch_dice_loss(x, x0)
    print('cost', cost_mask.shape)
    lsa = [linear_sum_assignment(m) for m in cost_mask]
    return lsa

def relabel_targets(x, x0):
    lsa = jnp.array(match_paths(x, x0))
    mapping = lsa[:, 1, :, None, None].expand(-1, -1, x0.shape[-2], x0.shape[-1])
    return jnp.take_along_axis(x0, mapping, axis=1)

def view_paths_matched(x, y0, x0):
    lsa = match_paths(x0, x)
    imgs = []
    y0 = y0
    for i in range(x.shape[0]):
        path = x[i,:,:,lsa[i][1]]
        tgt = x0[i,:,:,lsa[i][0]]
        depth = y0[i,:,:,1]
        mask = y0[i,:,:,0]
        mask = mask / np.max(mask)
        depth = depth / np.max(depth)
        img = np.stack([path, tgt, 0*tgt], axis=-1)
        img = np.hstack(img)
        img = np.hstack((np.dstack((depth*(mask>0), depth*(mask>0), mask)), np.dstack((jnp.max(x[i], axis=-1), jnp.max(x0[i], axis=-1), 0*jnp.max(x[i], axis=-1))), img))
        imgs.append(img)
    img = np.vstack(imgs)
    plt.imshow(img.astype(np.float32))
    return img

@ray.remote
def make_tangle_remote(i, s, np):
   return make_tangle(i, s, np)
   
def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    """
    smooth = 1.
    inputs = inputs.reshape((inputs.shape[0], -1, inputs.shape[-1]))
    targets = targets.reshape((targets.shape[0], -1, targets.shape[-1]))
    print(inputs.shape, targets.shape, inputs.dtype, targets.dtype)
    numerator = 2 * jnp.einsum("bin,bim->bnm", inputs, targets)
    print(numerator.shape)
    denominator = jnp.sum(inputs, axis=1)[:,:,None] + jnp.sum(targets, axis=1)[:,None,:]
    print(denominator.shape)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss



# Usage with batch processing for JAX
def get_batches(dataset, batch_size=32, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(dataset), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch = [dataset[i] for i in batch_indices]
        yield batch

import tensorflow as tf
import pickle
import ray

NP = 16
S = 64

class TangleDataset(tf.data.Dataset):
    """A custom TensorFlow Dataset class for the Tangle dataset."""
    def _generator(m, fn="", offset=0):
        # Generator function that yields data samples
        if fn:
            with open(fn, 'rb') as f:
                print('load')
                data = pickle.load(f)
                print('loaded', len(data))
        else:
            print('ray', m)
            data = ray.get([make_tangle_remote.remote(i+offset, S, NP) for i in range(m)])
            print('get data', len(data))

        for item in data:
            if item is not None:
                yield item

    def __new__(cls, m=10, fn="", offset=0):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32, tf.float32),  # Modify these types based on your actual data types
            output_shapes=(tf.TensorShape([S,S,NP]), tf.TensorShape([S,S,2])),  # Adjust these shapes based on your data
            args=(m,fn, offset)
        )


def prepare_for_training(ds, batch_size=32, shuffle_buffer_size=1000):
    # Shuffle the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat the dataset so each original value is seen once per epoch
    ds = ds.cache()
    #ds = ds.repeat()

    # Batch the data before training
    ds = ds.batch(batch_size)
    ds = ds.repeat()

    # Prefetch data for enhanced performance
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    def scale_and_reshape(xs):
        local_device_count = jax.local_device_count()
        def _scale_and_reshape(x):
           return x.reshape((local_device_count, -1) + x.shape[1:])

        xs = [v.numpy() for v in xs] 
        return jax.tree.map(_scale_and_reshape, xs)

    it = map(scale_and_reshape, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


# Example usage
m =  1024#*16
fn = None  # Specify filename if you have pre-saved data
bs = 4
t_bs = 2

tangle_dataset_a = TangleDataset(m)#, fn='train_100000.pkl')

tangle_dataset = prepare_for_training(tangle_dataset_a, batch_size=bs)

m = 160

test_dataset = TangleDataset(m, offset=1234567890)

test_dataset = prepare_for_training(test_dataset, batch_size=t_bs)


import math

class SinusoidalPosEmb(nn.Module):
    """Build sinusoidal embeddings 

    Attributes:
      dim: dimension of the embeddings to generate
      dtype: data type of the generated embeddings
    """
    dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, time):
        """
        Args:
          time: jnp.ndarray of shape [batch].
        Returns:
          out: embedding vectors with shape `[batch, dim]`
        """
        assert len(time.shape) == 1.
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=self.dtype) * -emb)
        emb = time.astype(self.dtype)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb



# Initialize wandb
#wandb.init(project="tangle-jax", entity="your-wandb-username")

class BasicBlock(nn.Module):
    in_channels: int
    out_channels: int
    temb_channels: int = 256
    stride: int = 1
    groups: int = 8

    @nn.compact
    def __call__(self, x, pos_emb, time_emb):
        identity = x

        B, _, _, C = x.shape

        x = x + pos_emb

        out = nn.Conv(self.out_channels, kernel_size=(7, 7), strides=(self.stride, self.stride), padding=[(3, 3), (3, 3)], name="conv1")(x)
        out = nn.LayerNorm(name="bn1")(out)
        out = jax.nn.relu(out)

        assert time_emb.shape[0] == B and len(time_emb.shape) == 2

        out = nn.GroupNorm(num_groups=self.groups,  name='norm_0')(out)

        # add in timestep embedding
        time_emb = nn.Dense(features=2 * self.out_channels,
                           name='time_mlp.dense_0')(nn.swish(time_emb))
        time_emb = time_emb[:,  jnp.newaxis, jnp.newaxis, :]  # [B, H, W, C]
        scale, shift = jnp.split(time_emb, 2, axis=-1)
        out = out * (1 + scale) + shift

        out = nn.swish(out)

        out = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", name="conv2")(out)
        out = nn.LayerNorm(name="bn2")(out)
        out = jax.nn.relu(out)


        out += identity
        return out
        
class DenoisingModel(nn.Module):
    NC: int = 128
    S: int = S  # Update with actual size

    @nn.compact
    def __call__(self, x, y, t):
    
        pos_emb_i = self.param('pos_emb_i', nn.initializers.normal(), (self.S, 1, self.NC//2))
        pos_emb_j = self.param('pos_emb_j', nn.initializers.normal(), (1, self.S, self.NC//2))

        batch_size = x.shape[0]
        expanded_shape_i = (batch_size, self.S, self.S, self.NC // 2)

        time_emb = SinusoidalPosEmb(self.NC)(t)  # [B. dim]
        time_emb = nn.Dense(features=self.NC, name='time_mlp.dense_0')(time_emb)



        #pos_emb = jnp.concatenate([jnp.broadcast_to(pos_emb_i.repeat(self.S, axis=1), expanded_shape_i),  jnp.broadcast_to(pos_emb_j.repeat(self.S, axis=0), expanded_shape_i)], axis=3)
        pos_emb = jnp.concatenate([jnp.broadcast_to(pos_emb_i[None], expanded_shape_i),  jnp.broadcast_to(pos_emb_j[None], expanded_shape_i)], axis=3)

        x = nn.Conv(self.NC, kernel_size=(1, 1), name="model_in")(x)
        y = nn.Conv(self.NC, kernel_size=(1, 1), name="model_in_maze")(y)


        pos_emb = y
        x = BasicBlock(self.NC, self.NC)(x, pos_emb, time_emb)
        x = BasicBlock(self.NC, self.NC)(x, pos_emb, time_emb)
        x = BasicBlock(self.NC, self.NC)(x, pos_emb, time_emb)
        x = BasicBlock(self.NC, self.NC)(x, pos_emb, time_emb)
        return nn.Conv(NP*2, kernel_size=(1, 1), name="model_out",   bias_init=jax.nn.initializers.zeros)(x)


import jax
import jax.numpy as jnp

def loss_fn(pred_x0_1, xt_1, x0_1, t, ddpm_params): #q_posterior, p_next):
    l = 1e-2
    posterior_x_t_m_1 = q_posterior(xt_1, x0_1, t, ddpm_params)
    pred_x_t_m_1 = p_next(xt_1, pred_x0_1, t, ddpm_params)
    simple = -(jnp.log(pred_x0_1) * x0_1).sum(axis=-1).mean()
    l_0 = -(jnp.log(pred_x_t_m_1) * x0_1).sum(axis=-1)
    l_KL = (posterior_x_t_m_1 * (jnp.log(posterior_x_t_m_1) - jnp.log(pred_x_t_m_1))).sum(axis=-1)
    vb = jnp.where(t[:, None, None, None] == 0, l_0, l_KL).mean()
    total = l * simple + vb
    return {'simple': simple, 'vb': vb, 'total': total}

class Loss:
    def __init__(self):
        self.l = 1e-2

    def loss_simple(self, pred_x0, x_0, t):
        """
        Computes a simple loss component.
        
        Args:
        pred_x0 (jax.numpy.ndarray): Predicted values.
        x_0 (jax.numpy.ndarray): True values.
        t (jax.numpy.ndarray): Time steps or condition.

        Returns:
        float: Computed simple loss.
        """
        return -(jnp.log(pred_x0) * x_0).sum(axis=-1).mean()

    def loss(self, posterior_x_t_m_1, pred_x_t_m_1, x0, t):
        """
        Computes the total loss including the KL divergence component.

        Args:
        posterior_x_t_m_1 (jax.numpy.ndarray): Posterior distribution estimates.
        pred_x_t_m_1 (jax.numpy.ndarray): Predicted next step values.
        x0 (jax.numpy.ndarray): Original data points.
        t (jax.numpy.ndarray): Time steps or condition.

        Returns:
        float: Combined loss including KL divergence.
        """
        l_0 = -(jnp.log(pred_x_t_m_1) * x0).sum(axis=-1)
        l_KL = (posterior_x_t_m_1 * (jnp.log(posterior_x_t_m_1) - jnp.log(pred_x_t_m_1))).sum(axis=-1)
        return jnp.where(t[:, None, None, None] == 0, l_0, l_KL).mean()

    def __call__(self, pred_x0_1, xt_1, x0_1, t, q_posterior, p_next):
        """
        Compute the total loss as a callable method of the class.
        
        Args:
        pred_x0_1 (jax.numpy.ndarray): Prediction from model.
        xt_1 (jax.numpy.ndarray): Intermediate values.
        x0_1 (jax.numpy.ndarray): Original values.
        t (jax.numpy.ndarray): Time steps or condition.
        q_posterior (function): Function to compute posterior distribution.
        p_next (function): Function to predict next values.

        Returns:
        dict: Dictionary containing all components of the loss.
        """
        posterior_x_t_m_1 = q_posterior(xt_1, x0_1, t)
        pred_x_t_m_1 = p_next(xt_1, pred_x0_1, t)
        simple = self.loss_simple(pred_x0_1, x0_1, t)
        vb = self.loss(posterior_x_t_m_1, pred_x_t_m_1, x0_1, t)
        total = self.l * simple + vb
        return {'simple': simple, 'vb': vb, 'total': total}


def q_posterior(xt, x0, t, ddpm_params):
    a = (x0 @ ddpm_params['Qt_prod'][t, None, None])
    b = jnp.sum(a * xt, axis=-1)[...,None]
    p = (xt @ ddpm_params['Qt'][t, None, None].mT) * (x0 @ ddpm_params['Qt_prod'][t-1, None, None])  / b
    return jnp.where(t[:,None,None,None,None]==0, x0, p)
    
def p_next(xt, pred_x0, t, ddpm_params):
    return jnp.where(t[:,None,None,None,None]==0, pred_x0, q_posterior(xt, pred_x0, t, ddpm_params))


# Define the loss function
def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

# Initialize the model and optimizer
def create_train_state(rng, learning_rate):
    model = DenoisingModel()
    params = model.init(rng, jnp.ones([1, S, S, 2*NP]), jnp.ones([1, S, S, 2]), jnp.ones([1]))['params']
    tx = optax.adam(learning_rate = 1e-4 , b1=0.9, b2 = 0.9999, 
            eps=1e-8)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def loss_simple(pred_x0, x_0):
        """
        Computes a simple loss component.
        
        Args:
        pred_x0 (jax.numpy.ndarray): Predicted values.
        x_0 (jax.numpy.ndarray): True values.
        t (jax.numpy.ndarray): Time steps or condition.

        Returns:
        float: Computed simple loss.
        """
        return -(jnp.log(pred_x0) * x_0).sum(axis=-1).mean()

from config import get_config

config = get_config()

def q_forward(x0, t, Qt_prod):

    p = jnp.einsum( 'b i j , b x y c i -> b x y c j', Qt_prod[t], x0)
    
    return p

def sample(x, uniform_noise):
    eps = 1e-6
    logits = jnp.log(x+eps)
    noise = jnp.clip(uniform_noise, 1e-6, 1.)
    gumbel_noise = - jnp.log(-jnp.log(noise))
    return jnp.argmax(logits + gumbel_noise, axis=-1)   

def ddpm_sample_step(rng, state, batch, ddpm_params):
    x0, y0 = batch
    b = x0.shape[0]
    rng, x0_rng = jax.random.split(rng)
    x = jax.random.randint(x0_rng, x0.shape, 0, 2)

    def step(i, p):
        x, rng = p
        t = 999 - i
        tv = jnp.array([t]*b) 
        x_1 = jax.nn.one_hot(x, 2)
        x_1f = x_1.reshape(*x_1.shape[:3], x_1.shape[3]*x_1.shape[4])
        pred = state.apply_fn({'params':state.params}, x_1f, y0, tv)
        pred = pred.reshape(*x_1.shape)
        pred = jax.nn.softmax(pred, axis=-1)
        rng, noise_rng = jax.random.split(rng)
        pred = p_next(x_1, pred, tv, ddpm_params)
        noise = jax.random.uniform(noise_rng, x_1.shape)
        x = sample(pred, noise)
        return x, rng
    
    x, rng = jax.lax.fori_loop(0, 999, step, (x, rng))   
    return x

# train step
def train_step(rng, state, batch, ddpm_params, loss_fn, pmap_axis='batch'):
    
    x0 = batch[0]
    y0 = batch[1]
    assert x0.dtype in [jnp.float32, jnp.float64]
    
   # create batched timesteps: t with shape (B,)
    B, H, W, C = x0.shape
    rng, t_rng = jax.random.split(rng)

    batched_t = jax.random.randint(t_rng, shape=(B,), dtype = jnp.int32, minval=0, maxval=1000)
  
    x0_1 = jax.nn.one_hot(x0, 2)

    # sample a noise (input for q_sample)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.uniform(noise_rng, x0_1.shape)

    xt_1 = q_forward(x0_1, batched_t, ddpm_params['Qt_prod'])
    noised_xt = sample(xt_1, noise)
    noised_xt_1 = jax.nn.one_hot(noised_xt, 2)

    xt_rearranged = noised_xt_1.reshape(*noised_xt_1.shape[:3], noised_xt_1.shape[3]*noised_xt_1.shape[4])

    target = x0_1#.reshape(*x0_1.shape[:3], x0_1.shape[3]*x0_1.shape[4])

    def compute_loss(params):
        pred = state.apply_fn({'params':params}, xt_rearranged, y0, batched_t)
        #loss = loss_fn(flatten(pred), flatten(target))
        pred = pred.reshape(*x0_1.shape)
        loss = loss_fn(jax.nn.softmax(pred), xt_1, target, batched_t, ddpm_params)
        return loss['total'].mean()
    
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function


    grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)

    gm = jax.tree.map(lambda x: jnp.sum(x*x), grads)

    new_state = state.apply_gradients(grads=grads)
    
    #pred = state.apply_fn({'params':state.params}, xt_rearranged, y0, batched_t)
    #gm = jnp.mean(jax.nn.softmax(pred), axis=(0,1,2))
    
    return new_state, loss, gm

def create_optimizer(config):
    if config.optimizer == 'Adam':
        optimizer = optax.adam(
            learning_rate = 3e-4 , b1=0.9, b2 = 0.9999, 
            eps=1e-8)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer
 
# Prepare your dataset (make sure it is loaded and preprocessed)
# Assume `dataset` is a tf.data.Dataset object prepared as described previously

# Convert TensorFlow tensors to numpy arrays for compatibility with JAX
def tf_to_numpy(batch):
    inputs, targets = batch
    return jnp.array(inputs), jnp.array(targets)

# Training loop
epochs = 1
batch_size = 32
learning_rate = 3e-4
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, learning_rate)
ddpm_params = utils.get_ddpm_params(config.ddpm)


import flax.serialization
import msgpack  # for binary serialization

# Save the state to a file
def save_state(state, filepath):
    with open(filepath, 'wb') as f:
        serialized_state = flax.serialization.to_bytes(state)
        f.write(serialized_state)


if jax.process_index() == 0:
    wandb.init(
        project='ddpm_jax',
        job_type='training')


from jax.tree_util import Partial
state = jax_utils.replicate(state)

train_step_p = Partial(train_step, ddpm_params=ddpm_params, loss_fn =loss_fn)
p_train_step = jax.pmap(train_step_p, axis_name = 'batch')

sample_step = functools.partial(ddpm_sample_step, ddpm_params=ddpm_params)
p_sample_step = jax.pmap(sample_step, axis_name='batch')

plt.figure()
plt.ion()


train_metrics = []
train_metrics_last_t = time.time()

for epoch in range(epochs):
    for i, batch in enumerate(pbar:=tqdm.tqdm(tangle_dataset)):  # Make sure your dataset yields batches as tuples
        
        rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        train_step_rng = jnp.asarray(train_step_rng)
        state, loss, gm = p_train_step(train_step_rng, state, batch)

        train_metrics.append(loss)


        if i%100==0:
            train_metrics = common_utils.get_metrics(train_metrics)

            summary = {
                'train/loss': jax.tree_map(lambda x: x.mean(), train_metrics)
            }
            summary['time/seconds_per_step'] =  (time.time() - train_metrics_last_t) /100

            train_metrics = []
            train_metrics_last_t = time.time()


            pbar.set_postfix({'loss': loss[0]})
            wandb.log({
                "train/step": i, ** summary
            })


            #        wandb.log({"epoch": epoch, "loss": loss.item()})


        if i%1000==0:
            batch = next(test_dataset)
            print(batch[0].shape)
            rng, *sample_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            sample_step_rng = jnp.asarray(sample_step_rng)

            x= p_sample_step(sample_step_rng, state, batch)
            img = view_paths_matched(x[0].astype(jnp.float32), batch[1][0], batch[0][0].astype(jnp.float32))
            img = (255*img/np.max(img)).astype(np.uint8)
#            plt.savefig(f'results_{i}.png')
            plt.pause(0.1)
            imwrite(f'results_{i}.png', img)
            wandb.log({
                "train/step": i, ** summary
            })
            sample_images = wandb.Image(img, caption = f'samples step {i}')
            wandb.log({'samples': sample_images})
            
        if i%10000==0:
            save_state(state, f'model_{i}.msgpack')
        if i==300000:
            break

# Finish the wandb run
#wandb.finish()

