import torch.utils.data as TorchData
from torch import nn, optim

from modules.model import Model
from modules.transformer import Transformer
from my_data_set import MyDataSet
from utils import greedy_decoder
from vocab_dict import VocabDict

## refer from https://blog.csdn.net/BXD1314/article/details/126187598

epochs = 100

## 训练机
sentences = [
    # encode input             decode input               decode ouput
    # P 代表填充，S代表开始符号，E 代表结束符号
    ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
    ['我 有 零 个 女 朋 友 P', 'S I have zero good friend .', 'I have zero girl friend . E'],
    ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E'],
]

## 测试集，记录在这里方便后续使用
## 输入：“我 有 一 个 女 朋 友”
## 输出：“i have a girlfriend"

enc_inputs, dec_inputs, dec_outputs = VocabDict.make_data(sentences)
loader = TorchData.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

transformer = Transformer().to(Model.device_type)
# 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(transformer.parameters(), lr=1e-3,
                      momentum=0.99)  # 用adam的话效果不好

# ====================================================================================================
for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        """
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
            Model.device_type), dec_inputs.to(Model.device_type), dec_outputs.to(Model.device_type)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(
            enc_inputs, dec_inputs)
        # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ==========================================================================================
# 预测阶段
# 测试集
sentences = [
    # enc_input                dec_input           dec_output
    ['我 有 零 个 女 朋 友 P', '', '']
]

enc_inputs, dec_inputs, dec_outputs = VocabDict.make_data(sentences)
test_loader = TorchData.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
enc_inputs, _, _ = next(iter(test_loader))

print()
print("=" * 30)
print("利用训练好的Transformer模型将中文句子'我 有 零 个 女 朋 友' 翻译成英文句子: ")
for i in range(len(enc_inputs)):
    greedy_dec_predict = greedy_decoder(transformer, enc_inputs[i].view(
        1, -1).to(Model.device_type), start_symbol=VocabDict.tgt_vocab["S"])
    print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
    print([VocabDict.src_idx2word[t.item()] for t in enc_inputs[i]], '->',
          [VocabDict.tgt_idx2word[n.item()] for n in greedy_dec_predict.squeeze()])