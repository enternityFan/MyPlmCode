# @Time : 2022-07-22 20:43
# @Author : Phalange
# @File : MLP.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.active = F.relu
        self.linear2 = nn.Linear(hidden_dim,output_dim)

    def forward(self,inputs):
        hidden = self.linear1(inputs)
        activation = self.active(hidden)
        outputs = self.linear2(activation)
        probs = F.softmax(outputs,dim=1)

        return probs



if __name__ == "__main__":
    clf = MLP(4,5,2)
    inputs = torch.rand(3,4)
    probs = clf(inputs)
    print(probs)