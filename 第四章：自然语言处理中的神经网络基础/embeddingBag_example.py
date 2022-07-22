# @Time : 2022-07-22 20:57
# @Author : Phalange
# @File : embeddingBag_example.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch
import torch.nn as nn


input1 = torch.tensor([0,1,2,1],dtype=torch.long)
input2 = torch.tensor([0,1,2,1,5],dtype=torch.long)
input3 = torch.tensor([0,1,1],dtype=torch.long)
input4 = torch.tensor([0,8,2,3,7],dtype=torch.long)

inputs = [input1,input2,input3,input4]
offsets = [0] + [i.shape[0] for i in inputs]
offsets = torch.tensor(offsets[:-1],dtype=torch.long).cumsum(dim=0)
inputs = torch.cat(inputs)



embeddingbag = nn.EmbeddingBag(num_embeddings=8,embedding_dim=3)
embeddings = embeddingbag(inputs,offsets)
print(embeddings)
print(embeddings.shape)



