# @Time : 2022-07-22 21:09
# @Author : Phalange
# @File : utils.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import torch
from torch.utils.data import DataLoader,Dataset
from Vocab import Vocab

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOW_TOKEN = "<bow>"
EOW_TOKEN = "<eow>"


WEIGHT_INIT_RANGE = 0.1


def load_reuters():
    from nltk.corpus import reuters
    text = reuters.sents()
    # lowercase (optional)
    text = [[word.lower() for word in sentence] for sentence in text]
    vocab = Vocab.build(text, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]

    return corpus, vocab



def save_pretrained(vocab,embeds,save_path):
    with open(save_path,"w") as writer:
        writer.write(f"{embeds.shape[0]}{embeds.shape[1]}\n")
        for idx,token in enumerate(vocab.idx_to_token):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")

        print(f"Pretrained embeddings saved to :{save_path}")




def load_pretrained(load_path):



    with open(load_path,'r') as fin:
        #n,d = map(int,fin.readline().split())
        tokens = []
        embeds = []
        for line in fin:
            line = line.rstrip().split(' ')

            token,embed = line[0],list(map(float,line[1:]))
            tokens.append(token)
            embeds.append(embed)

        vocab = Vocab(tokens)
        embeds = torch.tensor(embeds,dtype=torch.float)

    return vocab,embeds



def get_loader(dataset,batch_size,shuffle=True):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle
    )
    return data_loader

def init_weights(model):
    for name,param in model.named_parameters():
        if "embedding" not in name:
            torch.nn.init.uniform_(
                param,a=-WEIGHT_INIT_RANGE,b=WEIGHT_INIT_RANGE
            )