# @Time : 2022-07-22 20:51
# @Author : Phalange
# @File : emMLP.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.linear1 = nn.Linear(embedding_dim,hidden_dim)
        self.active = F.relu
        self.linear2 = nn.Linear(hidden_dim,output_dim)

    def forward(self,inputs):
        """
        这里就是进行了一个词袋模型。

        :param inputs:
        :return:
        """
        embeddings = self.embedding(inputs)
        embeddings = embeddings.mean(dim=1)
        hidden = self.linear1(embeddings)
        activation = self.active(hidden)
        outputs = self.linear2(activation)
        probs = F.softmax(outputs,dim=1)

        return probs



if __name__ == "__main__":
    clf = MLP(8,3,5,2)
    inputs = torch.tensor([[0,1,2,1],[4,6,6,7]],dtype=torch.long)
    probs = clf(inputs)
    print(probs)
