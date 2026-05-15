import uuid
import random
from datetime import date
# Config
class DPSOM_Config:
    def __init__(self):
        self.num_epochs_pretrain = 10
        self.num_epochs = 24
        self.batch_size = 300
        self.latent_dim = 32
        self.som_dim = (8, 8)
        self.learning_rate = 7e-4
        self.learning_rate_pretrain = 7e-4
        self.alpha = 10.0
        self.beta = 0.5
        self.gamma = 10.0
        self.theta = 0.1
        self.tau = 1.6   # Temporal smoothness weight
        self.eta = 1.0   # Record Attention weight
        self.delta_age = 1.0
        self.delta_sex = 1.0
        self.num_beats = 5
        self.decay_factor = 0.99
        self.decay_steps = 1000
        self.name = "ptb_T"
        self.ex_name = f"{self.name}_{self.latent_dim}_{self.som_dim[0]}-{self.som_dim[1]}_{str(date.today())}_{uuid.uuid4().hex[:5]}"
        self.logdir = f"logs/{self.ex_name}"
        self.modelpath = f"models/{self.ex_name}/{self.ex_name}.ckpt"
        self.data_set = "PTB-XL"
        self.dropout = 0.2
        self.prior_var = 1
        self.prior = 2.5
        self.use_saved_pretrain = False
        self.save_pretrain = False
        self.random_seed = random.SystemRandom().randint(0, 2**32 - 1)
        self.use_data_cache = False
