import ml_collections

def get_config():

  config = ml_collections.ConfigDict()

  # ddpm 
  config.ddpm = ddpm = ml_collections.ConfigDict()
  ddpm.beta_schedule = 'cosine'
  ddpm.timesteps = 1000
  ddpm.p2_loss_weight_gamma = 0. # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
  ddpm.p2_loss_weight_k = 0


  return config
