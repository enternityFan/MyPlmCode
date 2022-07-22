# @Time : 2022-07-08 15:11
# @Author : Phalange
# @File : main.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import numpy as np


tag2id,id2tag = {},{}
word2id,id2word = {},{}

for line in open("./traindata.txt"):
    items = line.split('/')
    word,tag = items[0],items[1].rstrip()


    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word

    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag

print(word2id)
print(tag2id)

M = len(word2id)
N = len(tag2id)


"""
构建PI A B
"""

pi = np.zeros(N)# 每个词性出现在句子中第一个位置的概率
A = np.zeros((N,M))# A[i][j] 给定tag i 出现单词j的概率，
B = np.zeros((N,N))# B[i][j] 之前的转态为i,之后转换为状态j的概率

prev_tag = ""
for line in open("traindata.txt"):
    items = line.split("/")
    wordId,tagId = word2id[items[0]],tag2id[items[1].rstrip()]
    if prev_tag == "": #现在是句子的开始
        pi[tagId] +=1
        A[tagId][wordId] +=1
    else:
         #如果不是句子的开始
        A[tagId][wordId] +=1
        B[tag2id[prev_tag]][tagId] +=1

    if items[0] == ".":
        prev_tag = ""
    else:
        prev_tag = items[1].rstrip()


pi = pi / sum(pi)#count to probably
for i in range(N):
    A[i] /=sum(A[i])
    B[i] /=sum(B[i])

# 到这里为之模型参数全部准备好了。


"""
dp[i][j] 代表了从w1到wi的最好的路径落在j个词性的最短路径，
dp[i][j] = max(dp[i-1]+A+B) + A[i][j]
"""

def predict(words):
    words = [each for each in words.split(" ")]
    T = len(words)
    dp = np.zeros((N,T))
    # 首先初始化第一行
    paths = np.zeros((N,T))

    for i in range(N):
        wordId = word2id[words[0]]
        dp[i][0] = A[i][wordId]

    # 开始计算全部的路径
    for i in range(1,T):
        wordId = word2id[words[i]]
        for j in range(N):

            max_score = -100000000
            max_idx = 0
            for k in range(N):
                score = dp[k][i-1] + np.log(A[j][wordId] + 0.00001) + np.log(B[k][j] + 0.00001)
                if score >=max_score:
                    max_score = score
                    max_idx = k
            dp[j][i] = max_score
            paths[j][i] = max_idx

    # 反推出来最大的路径的分数
    best_path = []

    for i in range(N):
        for j in range(T-1,-1,-1):
            dp[i][T-1] +=dp[int(paths[i][j])][j]
    best_score = np.max(dp[:,T-1])
    dp = dp.T
    paths = paths.T
    best_score_idx = np.argmax(dp[T-1])

    best_path = [0] * T
    best_path[T-1] = int(best_score_idx)
    for i in range(T-2,-1,-1):
        best_path[i] = paths[i+1][int(best_path[i+1])]


    # print(dp)
    # print(paths)
    # print(best_score)
    # print(best_score_idx)
    print(best_path)
    print(words)
    for i in range(len(best_path)):
        print(id2tag[best_path[i]],end=" ")


predict("Social Security number , passport number and details about the services provided for the payment")









