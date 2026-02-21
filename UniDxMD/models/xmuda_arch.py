import torch
import torch.nn as nn

from UniDxMD.models.resnet34_unet import UNetResNet34
from UniDxMD.models.spconv_unet import SpUNetBase
from UniDxMD.models.module.semantic_query import Semantic_query

import torch.nn.functional as F


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs,
                 codebook = None,
                 sigma = None
                 ):
        super(Net2DSeg, self).__init__()


        self.global_embedding = codebook

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs,codebook = self.global_embedding, num_classes = num_classes, sigma=sigma)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        self.mlp = nn.Sequential(
            nn.Linear(128, feat_channels),
            nn.ReLU(inplace=True),
        )

        # segmentation head
        self.cls_head1 = nn.Linear(feat_channels, num_classes)

        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.proj_head1 = nn.Linear(128, feat_channels)
        self.proj_head2 = nn.Linear(feat_channels * 2, feat_channels)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.cls_head2 = nn.Linear(feat_channels, num_classes)

        # 2D prototype
        self.proj_pty = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, self.global_embedding.embedding_dim))

        self.semantic_query = Semantic_query(transformer_width =self.global_embedding.embedding_dim)



    def forward(self, data_batch, feat_3d, prototype_2d = None, init = None):
        loss_mdd = None
        if self.training and not init and prototype_2d is not None:
            semantic_weight, entropy_loss = self.semantic_query(self.global_embedding, prototype_2d.float())
            loss_diff = F.kl_div(
                    F.log_softmax(self.global_embedding.weight, dim=1),
                    F.softmax(semantic_weight.detach(), dim=1),  
                    reduction='batchmean'
                                        )
            loss_mdd = loss_diff + entropy_loss

        else:
            loss_mdd = 0

        # (batch_size, 3, H, W)
        img = data_batch['img']

        hints = data_batch["depth"]
        hints = hints.to(img.device)
        # 2D network
        x, loss_csqm_2d = self.net_2d(img, hints)

        img_indices = data_batch['img_indices']
        img_feats = []
        for i in range(x.shape[0]):
            img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        img_feats = torch.cat(img_feats, 0)
        img_feats_fus = img_feats.clone().detach()
        img_feats_fus = torch.cat([img_feats_fus, feat_3d], dim=1)
        img_feats_fus = self.mlp(img_feats_fus)
        img_feats_fus = F.relu(self.proj_head1(img_feats_fus))  
        img_feats_fus = torch.cat([img_feats, img_feats_fus], dim=1)  
        img_feats_fus = self.proj_head2(img_feats_fus) 

        # linear
        x = self.cls_head1(img_feats_fus)
        proj_2d_pty = self.proj_pty(img_feats)
        img_feats_prob = F.softmax(x, dim=1)
        confidence, _ = torch.max(img_feats_prob, dim=1)

        preds = {
            'seg_logit': x,
            'confidence': confidence
        }

        if self.dual_head:
            preds['seg_logit2'] = self.cls_head2(img_feats)

        if self.training:
            return preds, loss_csqm_2d, loss_mdd, proj_2d_pty
        else:
            return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 codebook = None,
                 sigma = None,
                 use_color = False
                 ):
        super(Net3DSeg, self).__init__()


        self.codebook = codebook
        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = SpUNetBase(in_channels=1, codebook = self.codebook, num_classes = num_classes, sigma = sigma)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # 2nd segmentation head
        self.dual_head = dual_head
        self.use_color = use_color
        if self.use_color:
            self.linear_rgb_mask = nn.Linear(3, 1)

    def forward(self, data_batch):
        if self.use_color:
            mask_rgb = self.linear_rgb_mask(data_batch["x"][1])
            mask_rgb = torch.sigmoid(mask_rgb)
            data_batch["x"][1] = mask_rgb  # non-inplace operation


        inter_feat_3d, _ = self.net_3d.encoder_forward(data_batch['x'])
        x1, x2, feats, proj_3d_pty, loss_csqm_3d = self.net_3d.decoder_forward(inter_feat_3d)
        confidence = F.softmax(x1, dim=1)
        confidence, _ = torch.max(confidence, dim=1)

        preds = {
            'feats': feats,
            'seg_logit': x1,
            'confidence': confidence
        }

        if self.dual_head:
            preds['seg_logit2'] = x2


        if self.training:
            return preds, loss_csqm_3d, proj_3d_pty
        else:
            return preds



class JointPrototypeLearning_add_relation(nn.Module):
    def __init__(self, num_classes, feature_dim=128, momentum = 0.99):
        super(JointPrototypeLearning_add_relation, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=4,
            dropout=0.1
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.importance_weights = nn.Parameter(torch.ones(4) / 4)
        self.register_buffer("memory_bank_2d", torch.randn(num_classes, feature_dim))
        self.register_buffer("memory_bank_3d", torch.randn(num_classes, feature_dim))
        self.register_buffer("memory_bank_2d_src", torch.randn(num_classes, feature_dim))
        self.register_buffer("memory_bank_3d_src", torch.randn(num_classes, feature_dim))

        self.m = momentum  

    @torch.no_grad()
    def momentum_update_key_encoder_2d(self, feat, init=False):
        """
        Momentum update of the memory_bank
        """
        if init:
            self.memory_bank_2d = feat
        else:
            self.memory_bank_2d = self.memory_bank_2d * self.m + feat * (1. - self.m)

    @torch.no_grad()
    def momentum_update_key_encoder_3d(self, feat, init=False):

        if init:
            self.memory_bank_3d = feat
        else:
            self.memory_bank_3d = self.memory_bank_3d * self.m + feat * (1. - self.m)


    @torch.no_grad()
    def momentum_update_key_encoder_2d_src(self, feat, init=False):
        """
        Momentum update of the memory_bank
        """
        # print("----------init-----------:",init) # True # False # 
        if init:
            self.memory_bank_2d_src = feat
        else:
            self.memory_bank_2d_src = self.memory_bank_2d_src * self.m + feat * (1. - self.m)

    @torch.no_grad()
    def momentum_update_key_encoder_3d_src(self, feat, init=False):

        if init:
            self.memory_bank_3d_src = feat
        else:
            self.memory_bank_3d_src = self.memory_bank_3d_src * self.m + feat * (1. - self.m)



    def forward(self, source_2d_proto, source_3d_proto, target_2d_proto, target_3d_proto):

        source_2d_proto = source_2d_proto.detach().clone().requires_grad_(True)
        source_3d_proto = source_3d_proto.detach().clone().requires_grad_(True)
        target_2d_proto = target_2d_proto.detach().clone().requires_grad_(True)
        target_3d_proto = target_3d_proto.detach().clone().requires_grad_(True)

        all_protos = torch.stack([
            source_2d_proto, 
            source_3d_proto, 
            target_2d_proto, 
            target_3d_proto
        ], dim=0)  
        
        weights = F.softmax(self.importance_weights, dim=0)
        
        protos = all_protos.permute(1, 0, 2)  
        attn_output, _ = self.cross_attention(
            protos, 
            protos, 
            protos
        )  
        
        weighted_protos = (attn_output * weights.view(1, -1, 1)).sum(dim=1) 
        joint_protos = self.mlp(weighted_protos)  
        
        consistency_loss = self.compute_consistency_loss(
            all_protos, 
            joint_protos
        )
        
        return joint_protos, consistency_loss
    
    def compute_consistency_loss(self, all_protos, joint_protos):
        num_protos = all_protos.size(1)

        relation_matrices = F.cosine_similarity(
            all_protos.unsqueeze(2), 
            all_protos.unsqueeze(1),  
            dim=-1
        )  

        diff = relation_matrices.unsqueeze(0) - relation_matrices.unsqueeze(1)  
        diff = diff[torch.triu(torch.ones(4, 4), diagonal=1).bool()]  
        relation_loss = (diff ** 2).mean()

        sim = F.cosine_similarity(
            all_protos,  
            joint_protos.unsqueeze(0),  
            dim=-1
        )  
        proto_loss = (1 - sim).mean()

        return (proto_loss + relation_loss) / 2

    def get_aligned_prototypes(self, all_protos):
        with torch.no_grad():
            joint_protos, _ = self.forward(
                all_protos[0], 
                all_protos[1], 
                all_protos[2], 
                all_protos[3]
            )
        return joint_protos


def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)


if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
