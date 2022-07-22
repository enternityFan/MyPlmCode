# @Time : 2022-07-06 14:12
# @Author : Phalange
# @File : main.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import pandas as pd
import numpy as np
df = pd.read_excel("data/综合类中文词库.xlsx",header=None)
print("data read success!")

dic_words = set(df[0])# 保存词典库中读取的单词
word_prob = {"北京":0.03,"的":0.08,"天":0.005,"气":0.005,"天气":0.06,"真":0.04,"好":0.05,"真好":0.04,"啊":0.01,"真好啊":0.02,
             "今":0.01,"今天":0.07,"课程":0.06,"内容":0.06,"有":0.05,"很":0.03,"很有":0.04,"意思":0.06,"有意思":0.005,"课":0.01,
             "程":0.005,"经常":0.08,"意见":0.08,"意":0.01,"见":0.005,"有意见":0.02,"分歧":0.04,"分":0.02, "歧":0.005}

print (sum(word_prob.values()))


#  分数（10）
## TODO 请编写word_segment_naive函数来实现对输入字符串的分词
def word_segment_naive(input_str):
    """
    1. 对于输入字符串做分词，并返回所有可行的分词之后的结果。
    2. 针对于每一个返回结果，计算句子的概率
    3. 返回概率最高的最作为最后结果

    input_str: 输入字符串   输入格式：“今天天气好”
    best_segment: 最好的分词结果  输出格式：["今天"，"天气"，"好"]
    """

    # TODO： 第一步： 计算所有可能的分词结果，要保证每个分完的词存在于词典里，这个结果有可能会非常多。
    segments = []  # 存储所有分词的结果。如果次字符串不可能被完全切分，则返回空列表(list)

    # 格式为：segments = [["今天"，“天气”，“好”],["今天"，“天“，”气”，“好”],["今“，”天"，“天气”，“好”],...]
    def word_segment(start_index):
        """
        利用回溯的思想实现所有可能的分词结果
        """
        result = []
        if start_index >= len(input_str):  # 停止迭代
            return [""]
        else:
            for i in range(start_index + 1, len(input_str) + 1):
                word = input_str[start_index:i]
                if word in dic_words:
                    result = result + [word + ',' + x for x in word_segment(i)]
            print(result)
        return result

    results = word_segment(0)
    for result in results:
        segments.append(result.split(',')[:-1])
    print(segments)

    # TODO: 第二步：循环所有的分词结果，并计算出概率最高的分词结果，并返回
    best_segment = 0
    best_score = 10000000000
    for seg in segments:
        # TODO ...
        score = 0.0
        for word in seg:
            if word in word_prob:
                score += -np.log(word_prob[word])
            else:
                score += -np.log(0.000001)

        if score <= best_score:
            best_score = score
            best_segment = seg

    return best_segment

# 测试
print(word_segment_naive("北京的天气真好啊"))
print(word_segment_naive("今天的课程内容很有意思"))
print(word_segment_naive("经常有意见分歧"))