import torch
import torch.nn as nn
import math

# ScaledDotProductAttention 类实现了缩放点积注意力机制


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        # query, key, value 分别是查询、键和值，它们都是输入的张量
        d_k = query.size(-1)  # 计算键的维度
        # 使用矩阵乘法计算注意力分数，并除以 sqrt(d_k) 进行缩放
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 如果提供了掩码，则将掩码中的0对应的分数设置为一个非常小的数，这样在应用softmax时这些位置的权重会接近0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 使用softmax函数对分数进行归一化，得到注意力权重
        attention = torch.softmax(scores, dim=-1)
        # 使用注意力权重对值进行加权求和，得到输出
        output = torch.matmul(attention, value)
        return output, attention

# MultiHeadAttention 类实现了多头注意力机制


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 模型的维度
        self.d_k = d_model // heads  # 每个头的维度
        self.h = heads  # 头的数量

        # 为每个头创建线性变换层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # 输出层用于将多头的输出合并回原始维度
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 对查询、键和值应用线性变换，并重塑形状以适应多头机制
        query = self.q_linear(query).view(batch_size, -1, self.h, self.d_k)
        key = self.k_linear(key).view(batch_size, -1, self.h, self.d_k)
        value = self.v_linear(value).view(batch_size, -1, self.h, self.d_k)

        # 转置以准备多头注意力计算
        query, key, value = query.transpose(
            1, 2), key.transpose(1, 2), value.transpose(1, 2)

        # 调用 ScaledDotProductAttention 类计算注意力和输出
        scores, attention = ScaledDotProductAttention()(query, key, value, mask)

        # 将多头的输出合并回原始维度
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.out(concat)

        return output, attention

# FeedForward 类实现了前馈神经网络


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层

    def forward(self, x):
        x = self.linear1(x)  # 应用第一个线性变换
        x = torch.relu(x)  # 应用ReLU激活函数
        x = self.linear2(x)  # 应用第二个线性变换
        return x

# EncoderLayer 类实现了Transformer编码器的单层


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            heads, d_model)  # 多头注意力机制
        self.feed_forward = FeedForward(d_model, d_ff)  # 前馈网络

        # 归一化层，用于在注意力和前馈网络之后进行归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 多头注意力机制，不返回注意力权重
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        # 残差连接和归一化
        x = self.norm1(x + attn_output)
        # 前馈网络，然后进行残差连接和归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

# TransformerEncoder 类实现了Transformer编码器的多层堆叠


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, d_ff):
        super(TransformerEncoder, self).__init__()
        # 创建N个编码器层的列表
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff) for _ in range(N)])
        # 最终的归一化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 遍历所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
        # 应用最终的归一化
        return self.norm(x)


# 示例用法
d_model = 512  # 模型的维度
heads = 8  # 多头注意力的头数
d_ff = 2048  # 前馈网络的维度
N = 6  # 编码器层的数量
seq_length = 10  # 序列长度
batch_size = 32  # 批量大小

# 创建随机输入数据
x = torch.rand(batch_size, seq_length, d_model)

# 初始化Transformer编码器
encoder = TransformerEncoder(d_model, N, heads, d_ff)
print(encoder)
# 前向传播
output = encoder(x)
print(output.shape)  # 应该输出 [batch_size, seq_length, d_model]
print(output[0])
