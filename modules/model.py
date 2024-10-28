
class Model():
    device_type = 'cpu'
    src_len = 8
    tat_len = 7
    # embedding size, token 和位置编码的维度
    d_model = 512
    ## FeedForward dimension (两次线性层中的隐藏层 512 -> 2048 -> 512, 线性层是用来做特征提取的，当然最后会再加一个 project 层)
    d_ff = 2048
    d_k = d_v = 64 # dimension of K(=Q), V(Q和K的维度需要相同，这里为了方便让 K=V)
    n_layers = 6 # number of encoder of decoder layer (Block 的个数）
    n_heads = 8 # multi head 有几套头
