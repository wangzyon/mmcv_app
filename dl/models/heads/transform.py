import torch
from torch import nn
import math
from collections import OrderedDict
from ..builder import HEADS

__all__ = ["TransformerHead"]


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


#@save
def masked_softmax(X, valid_lens):
    """在X最后一个轴上掩蔽元素并执行softmax操作，一般在transformer decode中对key使用"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，hidden_dim)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # hidden_dim/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # hidden_dim/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # hidden_dim/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, hidden_dim, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，hidden_dim)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # hidden_dim/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # hidden_dim/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，hidden_dim)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class FFN(nn.Module):
    """前馈网络"""

    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """transformer编码器块"""

    def __init__(self,
                 attention_input_dim,
                 norm_shape,
                 ffn_input_dim,
                 ffn_hidden_dim,
                 num_heads,
                 dropout,
                 use_bias=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(attention_input_dim, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = FFN(ffn_input_dim, ffn_hidden_dim, attention_input_dim)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


@HEADS.register_module()
class TransformerHead(nn.Module):
    """transformer编码器，序列标注问题仅需要编码器部分，无需解码器"""

    def __init__(self,
                 input_dim,
                 norm_shape,
                 ffn_input_dim,
                 ffn_hidden_dim,
                 num_heads,
                 num_layers,
                 dropout,
                 use_bias=False,
                 **kwargs):
        """_summary_

        Args:
            input_dim (_type_): 每个q,k,v词嵌入映射的维度
            norm_shape (_type_): lay norm shape
            ffn_input_dim (_type_): 前馈神经网络输入
            ffn_hidden_dim (_type_): 前馈神经网络隐层
            num_heads (_type_): 多头注意力机制，头的数量
            num_layers (_type_): Transformer块的堆栈数量
            dropout (_type_): self-attention weight dropout or 残差shortcut前dropout
            use_bias (bool, optional): _description_. Defaults to False.
        """
        super().__init__(**kwargs)

        self.blks = nn.Sequential(
            OrderedDict([(f'block{i}',
                          EncoderBlock(input_dim, norm_shape, ffn_input_dim, ffn_hidden_dim, num_heads, dropout, use_bias))
                         for i in range(num_layers)]))

    def forward(self, X):
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, None)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
