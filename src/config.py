import uuid
import random
from datetime import date


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
        self.use_data_cache = True

        # --- Encoder architecture ---
        self.encoder_base_channels_1 = 32   # depthwise channels after first conv block
        self.encoder_base_channels_2 = 64   # depthwise channels after second conv block
        self.encoder_kernel_size = 7        # conv kernel size (stride fixed at 2)
        self.encoder_fc_hidden_dim = 512    # shared FC hidden dim after flattening

        # --- Demographic latent space (fraction of latent_dim) ---
        self.z_age_dim_factor = 0.25        # z_age_dim = int(latent_dim * factor)
        self.z_sex_dim_factor = 0.25        # z_sex_dim = int(latent_dim * factor)

        # --- Age-correction module ---
        self.age_corr_topk = 4              # TopK SOM nodes used for correction
        self.age_corr_lambda_max = 0.30     # maximum correction gate value
        self.age_corr_ramp_epochs = 10      # epochs over which gate ramps to max

        # --- SOM initialisation ---
        self.som_init_std = 0.05            # std for trunc-normal embedding init

        # --- Optimiser ---
        self.weight_decay = 0.0             # L2 regularisation for joint optimiser
        self.gradient_clip_norm = 5.0       # max-norm for gradient clipping
        self.lr_meta_factor = 5.0           # meta-branch lr = learning_rate / lr_meta_factor

        # --- Probe-fitting phases ---
        self.num_epochs_probe = 3           # epochs for age/sex linear probe fitting
        self.lr_probe = 1e-3                # learning rate for probe fitting
