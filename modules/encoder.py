from torch import nn

from modules.encoder_layer import EncoderLayer
from modules.model import Model
from modules.positional_encoding import PositionalEncoding
from utils import get_attn_pad_mask
from vocab_dict import VocabDict


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(VocabDict.src_vocab_size, Model.d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(
            Model.d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(Model.n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(
            enc_inputs)  # [batch_size, src_len, d_model]，将单个 token -> 根据字典转换成整数类型的 index -> 根据 embedding ,   转换成
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(
            0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵，用于将 'S' 'E' 这类的 pad token在计算加权平均是过滤掉。
        enc_self_attn_mask = get_attn_pad_mask(
            enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns