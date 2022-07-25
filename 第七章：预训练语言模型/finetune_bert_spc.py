# @Time : 2022-07-25 9:44
# @Author : Phalange
# @File : finetune_bert_spc.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import numpy as np
from datasets import load_dataset,load_metric
from transformers import BertTokenizerFast,BertForSequenceClassification,TrainingArguments,Trainer


dataset = load_dataset('glue','rte')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased',return_dict=True)
metric = load_metric('glue','rte')


# 对训练集进行分词
def tokenize(examples):
    return tokenizer(examples['sentence1'],examples['sentence2'],truncation=True,padding='max_length')

dataset = dataset.map(tokenize,batched=True)
encoded_dataset = dataset.map(lambda examples:{'labels':examples['label']},batched=True)

# 转换为torch.Tensor类型
columns = ['input_ids','token_type_ids','attention_mask','labels']
encoded_dataset.set_format(type='torch',columns=columns)




# 定义评价指标
def compute_metrics(eval_pred):
    predictions,labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions,axis=1),references=labels)

args = TrainingArguments(
    "ft-rte",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer = tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()