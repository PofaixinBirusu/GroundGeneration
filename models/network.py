import numpy as np
import open3d as o3d
import ghalton
import torch
from torch import nn
from utils import square_distance
from utils import ChamferLoss
from models.backbone_kpconv.blocks import block_decider
from models.transformer.position_embedding import PositionEmbeddingCoordsSine
from models.transformer.transformers import TransformerSelfEncoder, TransformerSelfEncoderLayer
from sklearn.metrics import precision_recall_fscore_support


class GGNet(nn.Module):
    def __init__(self, config):
        super(GGNet, self).__init__()
        self.config = config
        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim
        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.final_feats_dim = config.final_feats_dim

        ####################################################
        #           Make Encoder blocks
        ####################################################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(
                block, r,
                in_dim, out_dim,
                layer, config
            ))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
        ####################################################
        # Attention and Generate
        ####################################################
        self.feat_proj = nn.Linear(self.encoder_skip_dims[-1]*2, config.d_embed, bias=True)
        self.pos_embed = PositionEmbeddingCoordsSine(3, config.d_embed, scale=config.pos_emb_scaling)
        encoder_norm = nn.LayerNorm(config.d_embed) if config.pre_norm else None
        encoder_layer = TransformerSelfEncoderLayer(
            config.d_embed, config.nhead, config.d_feedforward, config.dropout,
            activation=config.transformer_act,
            normalize_before=config.pre_norm,
            sa_val_has_pos_emb=config.sa_val_has_pos_emb,
            ca_val_has_pos_emb=config.ca_val_has_pos_emb,
            attention_type=config.attention_type,
        )
        self.transformer_encoder = TransformerSelfEncoder(
            encoder_layer, config.num_encoder_layers, encoder_norm,
            return_intermediate=False
        )

        self.classify = nn.Sequential(
            nn.Linear(config.d_embed, config.d_embed),
            nn.ReLU(),
            nn.Linear(config.d_embed, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        feats = batch['features'].clone().detach()
        pts = batch['points'][-1]
        for block_i, block_op in enumerate(self.encoder_blocks):
            feats = block_op(feats, batch)
        pe = self.pos_embed(pts)
        pe = pe.unsqueeze(0)
        # print(feats.shape, pe.shape)
        feats = self.feat_proj(feats)
        feats_cond = self.transformer_encoder(
            feats.unsqueeze(0), feats.unsqueeze(0),
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            src_pos=pe if self.config.transformer_encoder_has_pos_emb else None,
            tgt_pos=pe if self.config.transformer_encoder_has_pos_emb else None
        )
        feats = feats_cond[0]
        score_pred = self.classify(feats)

        # ################### #仅用于画图  ############
        # overlaps = batch["ground_mask"]
        # nearest_ind = square_distance(batch["points"][-1].unsqueeze(0), batch["points"][0].unsqueeze(0))[0].min(dim=1)[1]
        # gts = [overlaps[0], overlaps[0], overlaps[0], overlaps[0][nearest_ind]]
        # for i in range(4):
        #     if i == 1 or i == 2:
        #         continue
        #     pts = batch["points"][i].detach().cpu().numpy()
        #     gt_idx = gts[i].detach().cpu().numpy()
        #
        #     pc = o3d.PointCloud()
        #     pc.points = o3d.Vector3dVector(pts)
        #     pc.colors = o3d.Vector3dVector(np.array([[0.7, 0.7, 0.7]]*pts.shape[0]))
        #     np.asarray(pc.colors)[gt_idx > 0.1] = np.array([0, 0.651, 0.929])
        #
        #     gt_pc = o3d.PointCloud()
        #     gt_pc.points = o3d.Vector3dVector(pts[gt_idx > 0.1])
        #     gt_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]]*np.asarray(gt_pc.points).shape[0]))
        #     o3d.draw_geometries([pc], window_name="level %d" % i, width=1000, height=800)
        #     o3d.draw_geometries([gt_pc], window_name="level %d" % i, width=1000, height=800)

        return score_pred, feats

    def compute_loss(self, score_pred, batch):
        nearest_ind = square_distance(batch["points"][-1].unsqueeze(0), batch["points"][0].unsqueeze(0))[0].min(dim=1)[1]
        gts = batch["ground_mask"][0][nearest_ind]

        def weighted_bce_loss(prediction, gt):
            loss = nn.BCELoss(reduction='none')
            gt = gt.float()
            class_loss = loss(prediction.view(-1), gt)

            weights = torch.ones_like(gt)
            w_negative = gt.sum() / gt.size(0)
            w_positive = 1 - w_negative

            weights[gt >= 0.5] = w_positive
            weights[gt < 0.5] = w_negative
            w_class_loss = torch.mean(weights * class_loss)

            #######################################
            # get classification precision and recall
            predicted_labels = prediction.detach().cpu().round().numpy()
            cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels, average='binary')
            return w_class_loss, cls_precision, cls_recall
        loss, precision, recall = weighted_bce_loss(score_pred, gts)
        return loss, precision, recall


# Ground Generation by Folding
class GGFoldingNet(nn.Module):
    def __init__(self, config, maxn=12000):
        super(GGFoldingNet, self).__init__()
        self.backbone = GGNet(config)
        self.folding = nn.Sequential(
            nn.Linear(config.d_embed+2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 3)
        )
        self.ratio = 1.
        self.cd = ChamferLoss()
        self.maxn = maxn

    def forward(self, batch, n=None):
        score_pred, feats = self.backbone(batch)
        pts = batch["points"][-1]
        select_inds = (score_pred.view(-1) > 0.5)
        # print(pts.shape, feats.shape, score_pred.shape)
        base_feats = feats[select_inds]
        if n is None:
            n = torch.sum(batch["ground_mask"][0].int(), dim=0).item()
            n = min(n, self.maxn)
        n_per_feat = n // base_feats.shape[0]
        # 在2维空间中采样均匀的[x, y], 且 0 < x, y < 1, 用离散的点来表示一个平面
        b_rnd = torch.tensor(ghalton.GeneralizedHalton(2).get(base_feats.shape[0]*n_per_feat), dtype=torch.float32)
        b_rnd = (b_rnd.to(pts.device) * self.ratio - self.ratio / 2) * 2
        # 平面折叠
        feat_num, feat_dim = base_feats.shape[0], base_feats.shape[1]
        ground_generation = self.folding(torch.cat([base_feats.view(feat_num, 1, feat_dim).repeat([1, n_per_feat, 1]).view(-1, feat_dim), b_rnd], dim=1))
        return ground_generation, score_pred

    def compute_loss(self, ground_generation, score_pred, batch):
        # 分类的 bce loss
        bce_loss, precision, recall = self.backbone.compute_loss(score_pred, batch)
        # 形状的 chamfer loss
        pts = batch["points"][0]
        ground_gt = pts[batch["ground_mask"][0].int() > 0.5]
        gtn = torch.sum(batch["ground_mask"][0].int(), dim=0).item()
        if ground_gt.shape[0] != gtn:
            print("check ground's shape")
        if gtn > self.maxn:
            ground_gt = ground_gt[torch.LongTensor(np.random.permutation(ground_gt.shape[0])[:self.maxn]).to(pts.device)]

        cd_loss = self.cd(ground_generation.unsqueeze(0), ground_gt.unsqueeze(0))
        return bce_loss, precision, recall, cd_loss


# Ground Generation by Coarse to Fine
class GGCFNet(nn.Module):
    def __init__(self, config, maxn=12000):
        super(GGCFNet, self).__init__()
        self.coarse = GGNet(config)
        self.fine = nn.Sequential(
            nn.Linear(config.d_embed+2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 3)
        )
        self.ratio = 1.
        self.cd = ChamferLoss()
        self.maxn = maxn

    def forward(self, batch, n=None):
        score_pred, feats = self.coarse(batch)
        pts = batch["points"][-1]
        select_inds = (score_pred.view(-1) > 0.5)
        # print(pts.shape, feats.shape, score_pred.shape)
        base_pts, base_feats = pts[select_inds], feats[select_inds]
        if n is None:
            n = torch.sum(batch["ground_mask"][0].int(), dim=0).item()
            n = min(n, self.maxn)
        n_per_feat = n // base_pts.shape[0]
        # 在2维空间中采样均匀的[x, y], 且 0 < x, y < 1, 用离散的点来表示一个平面
        b_rnd = torch.tensor(ghalton.GeneralizedHalton(2).get(base_pts.shape[0]*n_per_feat), dtype=torch.float32)
        b_rnd = (b_rnd.to(pts.device) * self.ratio - self.ratio / 2) * 2
        # 平面折叠
        feat_num, feat_dim = base_feats.shape[0], base_feats.shape[1]
        offsets = self.fine(torch.cat([base_feats.view(feat_num, 1, feat_dim).repeat([1, n_per_feat, 1]).view(-1, feat_dim), b_rnd], dim=1))
        ground_generation = base_pts.view(feat_num, 1, 3).repeat([1, n_per_feat, 1]).view(-1, 3) + offsets
        return ground_generation, score_pred

    def compute_loss(self, ground_generation, score_pred, batch):
        # 分类的 bce loss
        bce_loss, precision, recall = self.coarse.compute_loss(score_pred, batch)
        # 形状的 chamfer loss
        pts = batch["points"][0]
        ground_gt = pts[batch["ground_mask"][0].int() > 0.5]
        gtn = torch.sum(batch["ground_mask"][0].int(), dim=0).item()
        if ground_gt.shape[0] != gtn:
            print("check ground's shape")
        if gtn > self.maxn:
            ground_gt = ground_gt[torch.LongTensor(np.random.permutation(ground_gt.shape[0])[:self.maxn]).to(pts.device)]

        cd_loss = self.cd(ground_generation.unsqueeze(0), ground_gt.unsqueeze(0))
        return bce_loss, precision, recall, cd_loss
