# @Time : 2022-07-05 13:27
# @Author : Phalange
# @File : py3_1.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


"""
NLTK工具的学习

"""
import nltk
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.corpus import sentence_polarity
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentiwordnet
from nltk import pos_tag

nltk.download('sentence_polarity')
nltk.download('omw-1.4')
nltk.download('sentiwordnet')
#nltk.download()
print(stopwords.words('english'))

# for category in sentence_polarity.categories():
#     for sentence in sentence_polarity.sents(categories=category):
#         print(sentence,category)



"""
wordnet的使用
"""
syns = wordnet.synsets("bank")#返回bank的全部18个词义的synset
print(syns[0].name)#返回bank 第一个词义的名称，其中n表示名词
print(syns[0].definition())#返回bank第一个词义的定义，即河岸的定义
print(syns[1].definition())#返回第二个词义的定义，即银行的定义
print(syns[0].examples)#返回bank第一个词义的使用实例
print(syns[0].hypernyms)#返回bank第一个次以上的上位同义词集合
dog = wordnet.synset('dog.n.01')
cat = wordnet.synset('cat.n.01')
print(dog.wup_similarity(cat))#计算两个同义词集合之间的Wu-Palmer相似度


"""
SentiWordNet是情感倾向性词典
"""
print(sentiwordnet.senti_synset('good.a.01'))


"""
分句操作
"""
text = gutenberg.raw("austen-emma.txt")
sentences = sent_tokenize(text)# 对Emma 小说全文进行分句
print(sentences[100])#显示其中的一个句子

"""
标记(Token)解析 --- tokenization,其实就是分词的操作吧
"""
print(word_tokenize(sentences[100]))

"""
词性标注
"""
print(pos_tag(word_tokenize("They sat by the fire.")))#先进行了标记解析，然后在进行词性标注

print(pos_tag(word_tokenize("They fire a gun.")))

#关于词性标记的查询的功能
print(nltk.help.upenn_tagset('NN'))
print(nltk.help.upenn_tagset('VBP'))
print(nltk.help.upenn_tagset())#返回全部词性标记以及各词性的示例






