from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
import tinycudann as tcnn
from pointcept.datasets.sampart3d_util import *


@MODELS.register_module("SAMPart3D")
class SAMPart3D(nn.Module):

    def __init__(self,
                 backbone=None,
                 backbone_dim=None, 
                 output_dim=None,
                 pcd_feat_dim=None,
                 use_hierarchy_losses=True, 
                 max_grouping_scale=2, 
                 freeze_backbone=True,
                 **kwargs):
        super().__init__()

        self.use_hierarchy_losses = use_hierarchy_losses
        self.max_grouping_scale = max_grouping_scale
        self.device = "cuda"
        self.quantile_transformer = None

        self.backbone = build_model(backbone)
        self.init_feat = None

        self.instance_net = tcnn.Network(
            n_input_dims=backbone_dim+1,
            n_output_dims=output_dim,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 384,
                "n_hidden_layers": 6,
            },
        )

        self.pos_net = tcnn.Network(
            n_input_dims=pcd_feat_dim+1,
            n_output_dims=output_dim,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 384,
                "n_hidden_layers": 4,
            },
        )

        if freeze_backbone:
            for name, param in self.named_parameters():
                if 'instance_net' not in name and 'pos_net' not in name:
                    param.requires_grad = False

    def get_mlp(self, point_feat, scales):
        scales = self.quantile_transformer(scales)
        # n = point_feat.shape[0]
        point_feat = torch.cat((point_feat, scales), dim=-1)
        instance_pass = self.instance_net(point_feat)

        epsilon = 1e-5
        norms = instance_pass.norm(dim=-1, keepdim=True)
        instance_pass = instance_pass / (norms + epsilon)

        return instance_pass
    
    def pos_emb(self, point_feat, scales):
        scales = self.quantile_transformer(scales)
        # n = point_feat.shape[0]
        point_feat = torch.cat((point_feat, scales), dim=-1)
        instance_pass = self.pos_net(point_feat)

        epsilon = 1e-5
        norms = instance_pass.norm(dim=-1, keepdim=True)
        instance_pass = instance_pass / (norms + epsilon)

        return instance_pass

    def get_loss(self, input_dict, pcd_dict):
        if self.init_feat is None:
            with torch.no_grad():
                self.backbone.eval()
                point = self.backbone(pcd_dict)
                point_feat = point.feat
                self.init_feat = point_feat
                del self.backbone
        
        point_orgfeat_mapping = pcd_dict["feat"][input_dict["mapping"]]
        point_selected_feat = self.init_feat[input_dict["mapping"]]

        loss_dict = {}
        margin = 1.0

        ####################################################################################
        # Calculate GT labels for the positive and negative pairs
        ####################################################################################
        input_id1 = input_id2 = input_dict["mask_id"]

        # Expand labels
        labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
        labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)

        # Mask for positive/negative pairs across the entire matrix
        mask_full_positive = labels1_expanded == labels2_expanded
        mask_full_negative = ~mask_full_positive

        # Create a block mask to only consider pairs within the same image -- no cross-image pairs
        chunk_size = input_dict["nPxImg"]  # i.e., the number of rays per image
        num_chunks = input_id1.shape[0] // chunk_size  # i.e., # of images in the batch
        block_mask = torch.kron(
            torch.eye(num_chunks, device=self.device, dtype=bool),
            torch.ones((chunk_size, chunk_size), device=self.device, dtype=bool),
        )  # block-diagonal matrix, to consider only pairs within the same image
        
        # Only consider upper triangle to avoid double-counting
        block_mask = torch.triu(block_mask, diagonal=0)  
        # Only consider pairs where both points are valid (-1 means not in mask / invalid)
        block_mask = block_mask * (labels1_expanded != -1) * (labels2_expanded != -1)
        diag_mask = torch.eye(block_mask.shape[0], device=self.device, dtype=bool)
        scale = input_dict["scale"].view(-1, 1)

        ####################################################################################
        # Grouping supervision
        ####################################################################################
        total_loss = 0

        # 1. If (A, s_A) and (A', s_A) in same group, then supervise the features to be similar
        instance = self.get_mlp(point_selected_feat, scale)
        pose_emb = self.pos_emb(point_orgfeat_mapping, scale)
        instance = instance + pose_emb

        # instance = instance.float()
        mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
        instance_loss_1 = torch.norm(
            instance[mask[0]] - instance[mask[1]], p=2, dim=-1
        ).nan_to_num(0).mean()
        loss_weight_pos = torch.sum(mask_full_positive * block_mask * (~diag_mask)) / torch.sum(block_mask)
        total_loss += instance_loss_1 * loss_weight_pos

        # 2. If (A, s_A) and (A', s_A) in same group, then also supervise them to be similar at s > s_A
        if self.use_hierarchy_losses:
            scale_diff = torch.max(
                torch.zeros_like(scale), (self.max_grouping_scale - scale)
            )
            larger_scale = scale + scale_diff * torch.rand(
                size=(1,), device=scale.device
            )
            instance = self.get_mlp(point_selected_feat, larger_scale)
            pose_emb = self.pos_emb(point_orgfeat_mapping, larger_scale)
            instance = instance + pose_emb
            # instance = instance.float()
            mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
            instance_loss_2 = torch.norm(
                instance[mask[0]] - instance[mask[1]], p=2, dim=-1
            ).nan_to_num(0).mean()
            total_loss += instance_loss_2 * loss_weight_pos

        # 3. Also supervising A, B to be dissimilar at scales s_A, s_B respectively seems to help.
        instance = self.get_mlp(point_selected_feat, scale)
        pose_emb = self.pos_emb(point_orgfeat_mapping, scale)
        instance = instance + pose_emb
        # instance = instance.float()
        mask = torch.where(mask_full_negative * block_mask)
        instance_loss_3 = (
            F.relu(
                margin - torch.norm(instance[mask[0]] - instance[mask[1]], p=2, dim=-1)
            )
        ).nan_to_num(0).mean()
        loss_weight_neg = torch.sum(mask_full_negative * block_mask) / torch.sum(block_mask)
        total_loss += instance_loss_3 * loss_weight_neg


        loss_dict["instance_loss"] = total_loss
        loss_dict["instance_loss_1"] = instance_loss_1
        loss_dict["instance_loss_2"] = instance_loss_2
        loss_dict["instance_loss_3"] = instance_loss_3

        return loss_dict
    
    def forward(self, input_dict):
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.cuda()
                # print(k, v.shape)
        data_dict = input_dict["obj"]
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.cuda()
        data_dict["grid_size"] = 0.01
        if self.training:
            loss_dict = self.get_loss(input_dict, data_dict)
            return loss_dict
        else:
            if self.init_feat is None:
                with torch.no_grad():
                    self.backbone.eval()
                    point = self.backbone(data_dict)
                    point_feat = point.feat

            else:
                point_feat = self.init_feat

            scale = input_dict["scale"]
            n = data_dict["feat"].shape[0]
            scale_column = torch.full((n, 1), scale, device=point_feat.device)
            instance_feat = self.get_mlp(point_feat, scale_column)
            pose_emb = self.pos_emb(data_dict["feat"], scale_column)
            instance_feat = instance_feat + pose_emb
            
            return instance_feat

        
