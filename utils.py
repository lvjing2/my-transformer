import numpy as np
import torch

from modules.model import Model
from vocab_dict import VocabDict


## 1. 数据构建
def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    # [batch_size, 1, len_k], True is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # pad_attn_mask = [[[f,f,f,f,t]], [[f,f,f,f,t]]]
    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [[[],[],[],[],[],[],[],[]], [[],[],[],[],[],[],[],[]]]

def get_attn_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

def greedy_decoder(model, enc_input, start_symbol):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(Model.device_type), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(Model.device_type)],
                              -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == VocabDict.tgt_vocab["E"]:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [dec_input.to(Model.device_type), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(Model.device_type)],
    #     -1)
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict