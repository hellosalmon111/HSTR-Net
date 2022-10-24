import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
import time
from source.functions import *
from .trans_utils.transformer import SelfattentionOutLayer

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([])
        for i in range(4):
            self.linears.append(nn.Linear(d_model, d_model))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.size = d_model
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm(x)
        x = x + self.dropout(self.self_attn(x, x, x))
        return x + self.dropout(self.feed_forward(self.norm(x)))

class Backbone_Model(nn.Module):
    def __init__(self, config):
        super(Backbone_Model, self).__init__()

        model = models.resnet101(pretrained=False)
        parm = torch.load('backbone/resnet101-63fe2227.pth')
        model.load_state_dict(parm, strict=False)

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1[0]
        )
        self.conv1_3 = nn.Sequential(
            model.layer1[1:],
        )
        self.conv2_1 = nn.Sequential(
            model.layer2[0],
        )
        self.conv2_4 = nn.Sequential(
            model.layer2[1:],
        )
        self.others = nn.Sequential(
            model.layer3,
        )
        self.config = config

    def forward_features(self, x):
        output = self.features(x)
        output = self.conv1_3(output)
        output = self.conv2_1(output)
        output = self.conv2_4(output)
        output = self.others(output)
        return output

    def forward(self, x):
        x = self.forward_features(x)
        return x

class Decoupling_Model(nn.Module):
    def __init__(self, config, num_AU_points):
        super(Decoupling_Model, self).__init__()

        self.num_classes = num_AU_points
        self.dec_conv1 = nn.Conv2d(in_channels=config.dim, out_channels=config.dim, kernel_size=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=config.dim, momentum=config.momentum)
        self.dec_conv2 = nn.Conv2d(in_channels=config.dim, out_channels=config.dim, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=config.dim, momentum=config.momentum)
        self.dec_conv3 = nn.Conv2d(in_channels=config.dim, out_channels=num_AU_points, kernel_size=1)
        self.dec_feat_conv1 = nn.Conv2d(in_channels=config.dim, out_channels=config.dim, kernel_size=1)
        self.dec_feat_bn1 = nn.BatchNorm2d(num_features=config.dim, momentum=config.momentum)
        self.dec_feat_conv2 = nn.Conv2d(in_channels=config.dim, out_channels=num_AU_points, kernel_size=1)
        self.pred_linear = nn.Linear(self.num_classes, self.num_classes)

    def forward(self, input):
        x = self.dec_conv1(input)
        x = F.relu(self.dec_bn1(x))
        x = self.dec_conv2(x)
        x = F.relu(self.dec_bn2(x))
        x = self.dec_conv3(x)
        x = x.view(x.size(0), self.num_classes, -1)
        x = F.softmax(x, dim=2)
        dec_map = x.view([-1, self.num_classes, 14, 14])
        y = self.dec_feat_conv1(input)
        y = F.relu(self.dec_feat_bn1(y))
        conf_map = self.dec_feat_conv2(y)
        dec_map = torch.mul(conf_map, dec_map)
        pred_decoupling = dec_map.mean(3).mean(2)
        pred_decoupling = self.pred_linear(pred_decoupling)
        return dec_map, pred_decoupling

class HSTR_AU(nn.Module):
    def __init__(self, config, num_AU_points, corr_path):
        super(HSTR_AU, self).__init__()

        self.config = config
        self.num_classes = num_AU_points
        self.T_lenth = config.T_lenth
        self.N = 3
        self.e_net = Backbone_Model(config)
        self.BN = nn.BatchNorm2d(num_features=1024, momentum=0.999)
        self.d_net = Decoupling_Model(config, num_AU_points)
        self.relu = nn.LeakyReLU(0.2)
        self.gcns = nn.ModuleList([])
        self.gcnt = nn.ModuleList([])
        gcs = GraphConvolution(196, 196) # 1024 1024
        gct = GraphConvolution(196, 196) # 1024 1024
        for _ in range(self.N):
            self.gcns.append(gcs)
            self.gcnt.append(gct)
        _adj = gen_A(num_AU_points, config.t, corr_path)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.A_T = Parameter(torch.from_numpy(
            np.diagflat(np.ones(self.T_lenth - 1, int), 1) + np.eye(self.T_lenth) + np.diagflat(
                np.ones(self.T_lenth - 1, int), 1).T).float())
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(392, self.num_classes, 1)
        self.glo_linear = nn.Linear(self.T_lenth, 1)
        dropout = 0
        d_model = 196
        d_ff = d_model * 3
        h = config.num_heads
        layers = EncoderLayer(d_model, h, d_ff, dropout)
        layert = EncoderLayer(d_model, h, d_ff, dropout)
        self.encoders = nn.ModuleList([])
        self.encodert = nn.ModuleList([])
        for _ in range(self.N):
            self.encoders.append(layers)
            self.encodert.append(layert)
        self.att_outlayer = SelfattentionOutLayer(d_model*2, 4, dim_feedforward=d_model*6, dropout=0,
                 activation="gelu", normalize_before=False)

    def forward_feature(self, x):
        x = self.features(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_4(x)
        x = self.others(x)
        return x

    def forward_st_dgcn(self, x):
        B, T, C, D = x.shape
        adjs = gen_adj(self.A).detach()
        adgt = self.A_T.detach()

        for i in range(self.N):
            x = x.contiguous().view(B*T, C, D)
            x = self.gcns[i](x, adjs)
            x = self.relu(x)
            x = x.view(B, T, C, D).transpose(1, 2).contiguous().view(B*C, T, D)
            x = self.gcnt[i](x, adgt)
            x = self.relu(x)
            x = x.view(B, C, T, D).transpose(1, 2)
        return x

    def forward_st_trans(self, x, mask=None):
        B, T, C, D = x.shape
        for i in range(self.N):
            x = x.contiguous().view(B * T, C, D)
            x = self.encoders[i](x, mask)
            x = x.view(B, T, C, D).transpose(1, 2).contiguous().view(B * C, T, D)
            x = self.encodert[i](x, mask)
            x = x.view(B, C, T, D).transpose(1, 2)
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.e_net(x)
        pred_global = self.BN(x)
        feature_map, out1 = self.d_net(pred_global)
        out1 = out1.view(B, T, -1).max(1)[0]
        v = feature_map.view(feature_map.size(0), feature_map.size(1), -1)
        v = v.view(B, T, self.num_classes, -1)
        z1 = self.forward_st_dgcn(v)
        z2 = self.forward_st_trans(v)
        z = torch.cat((z1, z2), dim=3)
        z = self.att_outlayer(z)
        out2 = self.last_linear(z)
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        return out2, out1

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.e_net.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.e_net.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
            ]

class Fusion_Model(nn.Module):
    def __init__(self, config):
        super(Fusion_Model, self).__init__()
        self.fc = nn.Linear(config.num_AU_points, config.num_AU_points)

    def forward(self, x):
        x = self.fc(x)
        return x
