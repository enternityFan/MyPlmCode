# @Time : 2022-07-05 8:58
# @Author : Phalange
# @File : py2_2.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D




def fmm_word_seg(sentence,lexicon,max_len):
    """
    正向最大匹配
    :param sentence: 待分词的句子
    :param lexicon: 词典
    :param max_len: 词典中最长单词长度
    :return:
    """

    begin = 0
    end = min(begin+max_len,len(sentence))
    words = []
    while begin < end:
        word = sentence[begin:end]
        if word in lexicon or end - begin == 1:
            words.append(word)
            begin = end
            end = min(begin+max_len,len(sentence))
        else:
            end -=1
    return words



def load_dict():
    f = open("lexicon.txt",encoding='utf-8')
    lexicon = set()
    max_len = 0
    for line in f:
        word = line.strip()
        lexicon.add(word)
        lexicon.add(word)
        if len(word) > max_len:
            max_len = len(word)
    f.close()

    return lexicon,max_len

lexicon,max_len = load_dict()
words = fmm_word_seg(input("请输入一个句子:"),lexicon,max_len)

for word in words:
    print(word,)

