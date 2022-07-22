# @Time : 2022-07-06 8:30
# @Author : Phalange
# @File : py3_2.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

"""
LTP的学习
好吧，这个不学了，我看网上说这个安装很复杂的。。。还得指定python3.5 或者3.6，还得这还得那的
"""


from ltp import LTP

ltp = LTP()#默认加载SMALL模型，首次使用的时候就会自动下载并且加载模型
segment,hidden = ltp.seg(["南京市长江大桥。"])#对巨资进行分词，结果使用segment访问，hidden用来访问每个词的隐含层向量，用于后续的分析的步骤
print(segment)#这个分词结果还是比较正常的，没分为  南京，市长.



