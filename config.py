import torch


class Config:
    def __init__(self):
        self.num_layers = 4
        self.in_points_dim = 3
        self.first_feats_dim = 128
        self.final_feats_dim = 96
        self.first_subsampling_dl = 0.025
        self.in_feats_dim = 1
        self.conv_radius = 2.5
        self.deform_radius = 5.0
        self.num_kernel_points = 15
        self.KP_extent = 2.0
        self.KP_influence = "linear"
        self.aggregation_mode = "sum"
        self.fixed_kernel_points = "center"
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.02
        self.deformable = False
        self.modulated = False
        self.neighborhood_limits = [40, 40, 40, 40]

        self.batch_size = 1
        self.num_workers = 4

        self.max_epoch = 40
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.architecture = [
            'simple',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb',
            'resnetb_strided',
            'resnetb',
            'resnetb'
        ]

        self.d_embed = 128
        self.pre_norm = True
        self.num_encoder_layers = 3
        self.nhead = 2
        self.d_feedforward = 512
        self.transformer_act = "relu"
        self.sa_val_has_pos_emb = True
        self.ca_val_has_pos_emb = True
        self.transformer_encoder_has_pos_emb = True
        self.attention_type = "dot_prod"
        self.dropout = 0.0
        self.pos_emb_scaling = 1.0
