# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
from torch import nn

from modules.model import Model

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(Model.d_model, Model.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(Model.d_ff, Model.d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return nn.LayerNorm(Model.d_model).to(Model.device_type)(output + residual)