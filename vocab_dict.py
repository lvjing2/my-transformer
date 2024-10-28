import torch


## 建立单词库，中文和英文单词库要分开建立
class VocabDict():

    src_vocab = {'P': 0, '我': 1, '有': 2, '一': 3, '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}
    src_idx2word = {i: w for i, w in enumerate(src_vocab)}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4, 'friend': 5, 'zero': 6, 'girl': 7, 'boy': 8, 'S': 9,
                 'E': 10, '.': 11}
    tgt_idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    def __init__(self):
        self.__init__()

    @classmethod
    def make_data(cls, sentences):
        """把单词序列转换为数字序列"""
        enc_inputs, dec_inputs, dec_outputs = [], [], []

        for i in range(len(sentences)):
            enc_input = [[cls.src_vocab[n] for n in sentences[i][0].split()]]
            dec_input = [[cls.tgt_vocab[n] for n in sentences[i][1].split()]]
            dec_output = [[cls.tgt_vocab[n] for n in sentences[i][2].split()]]

            # [[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
            enc_inputs.extend(enc_input)
            # [[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
            dec_inputs.extend(dec_input)
            # [[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
            dec_outputs.extend(dec_output)

        return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

