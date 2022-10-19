from torch import nn
from ..builder import BACKBONES


@BACKBONES.register_module()
class BilstmEncoder(nn.Module):

    def __init__(self, element_num, embedding_dim, hidden_dim, num_layers, only_last_hidden=True):
        """
        基于Bilstm的序列点特征提取  
        
        Args:
            element_num (int): 序列点在词库中的索引
            embedding_dim (int): 词向量编码长度
            hidden_dim (int): lstm隐层输出维度
            num_layers (int): lstm块堆栈层数
            only_last_hidden: (bool, optional): only_last:`False`输出bilstm所有隐层; only_last_hidden:`True`输出bilstm最后一个隐层
        """
        super().__init__()
        self.embedding = nn.Embedding(element_num, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.only_last_hidden = only_last_hidden

    def forward(self, x):
        self.lstm.flatten_parameters()
        embeddings = self.embedding(x)
        hiddens, _ = self.lstm(embeddings)
        if self.only_last_hidden:
            hiddens = hiddens[:, -1, :]
        return hiddens
