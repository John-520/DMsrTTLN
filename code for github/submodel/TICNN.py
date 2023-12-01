#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings



# ----------------------------inputsize >=28-------------------------------------------------------------------------
class TICNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(TICNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")
            
        
        
        self.feature_layers1 = nn.Sequential(
            ########   81 = 5+4*19   ;11+10*7;    21+20*3
            nn.Conv1d(1, 16, kernel_size=64, stride=8,padding=1,dilation=1),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
            )
        self.feature_layers2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        self.feature_layers3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        self.feature_layers4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.7),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        self.feature_layers5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.7),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        self.feature_layers6 = nn.Sequential(
            nn.Conv1d(64, 1024, kernel_size=3, stride=1),  # 32, 24, 24
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),     
            )
        self.pooling_layers = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool1d(4)  # 128, 4,4
            )
        self.__in_features = 1024
        # self.CBAMnet = CBAMLayer(self.__in_features)
        
    def forward(self, x):
        x = self.feature_layers1(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        
        x = self.feature_layers2(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        
        x = self.feature_layers3(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        
        x = self.feature_layers4(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        
        x = self.feature_layers5(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        
        x = self.feature_layers6(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        
        x = self.pooling_layers(x)
        
        'CBAM'
        # attention = MultiHeadAttention(x.shape[0], x.shape[2], x.shape[2], x.shape[2], 5, 0.5)
        # ans = attention(x,x,x)
        
        # config = {
        #     "num_of_attention_heads": 1,
        #     "hidden_size": x.shape[2]    }
        # selfattn = BertSelfAttention(config)
        # x = selfattn(x)
        
        
        # x = x.view(x.size(0), 1, x.size(1)*x.size(2))
        # multihead_attn = nn.MultiheadAttention(x.shape[2], 2)
        # x, attn_output_weights = multihead_attn(x, x, x)
        
        return x


    def output_num(self):
        return self.__in_features





# batch_size, num_queries, num_hiddens, num_heads  = 2, 4, 100, 5

# print(ans.shape)


















import torch
import torch.nn as nn
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1).cuda()
        self.avg_pool = nn.AdaptiveAvgPool1d(1).cuda()
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        ).cuda()
        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
    
    
    
    
    
    






        # self.feature_layers = nn.Sequential(
        #     ########   81 = 5+4*19   ;11+10*7;    21+20*3
        #     nn.Conv1d(1, 16, kernel_size=64, stride=8,padding=1,dilation=1),  # 16, 26 ,26
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(inplace=True),
        #    # nn.Dropout(0.5),
        #     nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
            
        #     nn.Conv1d(16, 32, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(inplace=True),
        #   #  nn.Dropout(0.5),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
            
        #     nn.Conv1d(32, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #    # nn.Dropout(0.5),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
            
        #     nn.Conv1d(64, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #   #  nn.Dropout(0.7),
        #     nn.MaxPool1d(kernel_size=2, stride=2),

        #     nn.Conv1d(64, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #   #  nn.Dropout(0.7),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
            
        #     nn.Conv1d(64, 1024, kernel_size=3, stride=1),  # 32, 24, 24
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #   #  nn.RReLU(0.0,0.5),
        #   #  nn.SELU(inplace=True),
        #   #  nn.Dropout(0.5),     
        #     nn.MaxPool1d(kernel_size=2, stride=2)
        #    # nn.AdaptiveMaxPool1d(4)  # 128, 4,4
           
        #     )
        # self.__in_features = 1024






import math

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class SelfAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `queries` 的形状：(`batch_size`, 查询的个数, `d`)
    # `keys` 的形状：(`batch_size`, “键－值”对的个数, `d`)
    # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
    # `valid_lens` 的形状: (`batch_size`,) 或者 (`batch_size`, 查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

# attention = SelfAttention(dropout=0.5)
# batch_size, num_queries, num_hiddens  = 2, 4, 10
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                  num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# batch_size, num_queries, num_hiddens, num_heads  = 1, 4, 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans.shape)


# batch_size, num_queries, num_hiddens, num_heads  = 2, 4, 100, 5
# attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
# X = torch.ones((batch_size, num_queries, num_hiddens))
# ans = attention(X, X, X)
# print(ans.shape)


















import torch
import torch.nn as nn

import math

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["hidden_size"] % config[
            "num_of_attention_heads"] == 0, "The hidden size is not a multiple of the number of attention heads"

        self.num_attention_heads = config['num_of_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_of_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)

        return output


# if __name__ == "__main__":
#     config = {
#         "num_of_attention_heads": 2,
#         "hidden_size": 20
#     }

#     selfattn = BertSelfAttention(config)
#     # print(selfattn)
#     embed_rand = torch.rand((1, 3, 20))
#     print(f"Embed Shape: {embed_rand.shape}")
#     # print(f"Embed Values:\n{embed_rand}")

#     output = selfattn(embed_rand)
#     print(f"Output Shape: {output.shape}")
#     # print(f"Output Values:\n{output}")
    

