# model/TRAR/trar.py

from model.TRAR.fc import MLP
import copy
from model.TRAR.layer_norm import LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class SoftRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        logits = self.mlp(x)
        alpha = F.gumbel_softmax(logits, tau=tau, hard=False)
        y = torch.bmm(alpha.unsqueeze(1), masks.float())
        return y, alpha

class MHA(nn.Module):
    def __init__(self, __C, att, v_size, q_size, k_size, dropout):
        super(MHA, self).__init__()
        self.__C = __C
        self.att = att
        self.linear_v = nn.Linear(v_size, att['hidden_size_v'] * att['multi_head'])
        self.linear_k = nn.Linear(k_size, att['hidden_size_k'] * att['multi_head'])
        self.linear_q = nn.Linear(q_size, att['hidden_size_q'] * att['multi_head'])
        self.linear_merge = nn.Linear(att['hidden_size_v'] * att['multi_head'], v_size)
        self.dropout = nn.Dropout(dropout)
        # The routing block is part of the MHA, as confirmed by the error
        self.routing_block = SoftRoutingBlock(__C["hidden_size"], __C.get("orders", 4))

    def forward(self, v, k, q, mask, get_alpha=False):
        n_batches = q.size(0)
        v_ = self.linear_v(v).view(n_batches, -1, self.att['multi_head'], self.att['hidden_size_v']).transpose(1, 2)
        k_ = self.linear_k(k).view(n_batches, -1, self.att['multi_head'], self.att['hidden_size_k']).transpose(1, 2)
        q_ = self.linear_q(q).view(n_batches, -1, self.att['multi_head'], self.att['hidden_size_q']).transpose(1, 2)
        
        att_mask = mask
        alpha = None

        if get_alpha:
            masks = getMasks(mask, self.__C)
            print('DEBUG: k shape before pooling for router:', k.shape)
            pooled_k = k.mean(1)  # [batch, hidden_size]
            route, alpha = self.routing_block(pooled_k, self.__C["tau_max"], masks)
            att_mask = route

        atted = self.att_func(v_, k_, q_, att_mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.att['hidden_size_v'] * self.att['multi_head'])
        atted = self.linear_merge(atted)

        if get_alpha:
            return atted, alpha
        return atted

    def att_func(self, v, k, q, mask):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask.unsqueeze(1).expand_as(scores), -1e9)
            else:
                # Assume mask is float (soft routing): multiply after softmax
                att_prob = F.softmax(scores, dim=-1)
                att_prob = att_prob * mask.unsqueeze(1)
                att_prob = self.dropout(att_prob)
                return torch.matmul(att_prob, v)
        att_prob = F.softmax(scores, dim=-1)
        att_prob = self.dropout(att_prob)
        return torch.matmul(att_prob, v)


class FFN(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(FFN, self).__init__()
        self.mlp = MLP(
            input_dim=in_size,
            hidden_dim=mid_size,
            output_dim=out_size,
            dropout=dropout_r,
            activation="ReLU" if use_relu else None
        )
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        return self.dropout(self.mlp(x))


class multiTRAR_SA_block(nn.Module):
    def __init__(self, __C):
        super(multiTRAR_SA_block, self).__init__()
        self.__C = __C
        att_config = {
            'multi_head': __C['multihead'],
            'hidden_size_v': __C['hidden_size'] // __C['multihead'],
            'hidden_size_k': __C['hidden_size'] // __C['multihead'],
            'hidden_size_q': __C['hidden_size'] // __C['multihead']
        }
        self.mhatt1 = MHA(__C, att_config, __C['hidden_size'], __C['hidden_size'], __C['hidden_size'], __C['dropout'])
        self.mhatt2 = MHA(__C, att_config, __C['hidden_size'], __C['hidden_size'], __C['hidden_size'], __C['dropout'])
        self.ffn = FFN(__C["hidden_size"], __C["ffn_size"], __C["hidden_size"], __C["dropout"], True)
        self.norm1 = LayerNorm(__C['hidden_size'])
        self.norm2 = LayerNorm(__C['hidden_size'])
        self.norm3 = LayerNorm(__C['hidden_size'])
        self.dropout1 = nn.Dropout(__C['dropout'])
        self.dropout2 = nn.Dropout(__C['dropout'])
        self.dropout3 = nn.Dropout(__C['dropout'])

    def forward(self, y, x, y_mask, x_mask):
        y = self.norm1(y + self.dropout1(self.mhatt1(y, y, y, y_mask)))
        
        cross_attention_out, alpha = self.mhatt2(x, x, y, x_mask, get_alpha=True)
        y = self.norm2(y + self.dropout2(cross_attention_out))
        
        y = self.norm3(y + self.dropout3(self.ffn(y)))
        return y, alpha


def getMasks(x_mask, __C):
    b = x_mask.shape[0]
    n = __C['IMG_SCALE'] * __C['IMG_SCALE']
    orders = __C.get("orders", 4)
    return torch.ones(b, orders, n, dtype=torch.bool).to(x_mask.device)


class DynRT_ED(nn.Module):
    def __init__(self, opt):
        super(DynRT_ED, self).__init__()
        self.opt = opt
        self.dec_list = nn.ModuleList([
            multiTRAR_SA_block(self._get_layer_opt(opt, i))
            for i in range(opt['layer'])
        ])

    def _get_layer_opt(self, opt, i):
        layer_opt = copy.deepcopy(opt)
        if 'ORDERS' in opt:
            layer_opt['orders'] = opt['ORDERS'][i]
        return layer_opt

    def forward(self, y, x, y_mask, x_mask):
        all_alphas = []
        for i in range(self.opt['layer']):
            y, alpha = self.dec_list[i](y, x, y_mask, x_mask)
            all_alphas.append(alpha)
        return y, x, all_alphas