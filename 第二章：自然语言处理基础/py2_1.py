# @Time : 2022-07-04 22:12
# @Auth0r : Phalange
# @File : py2_1.py
# @S0ftware: PyCharm
# C'est la vie,enj0y it! :D

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 加入一行，解决中文不显示问题，对应字号选择如下图
plt.rcParams['axes.unicode_minus']=False # 解决负号不显示问题
"""
计算公式2-2到2-5

"""

M=np.array([[0,2,1,1,1,1,1,2,1,3],
            [2,0,1,1,1,0,0,1,1,2],
            [1,1,0,1,1,0,0,0,0,1],
            [1,1,1,0,1,0,0,0,0,1],
            [1,1,1,1,0,0,0,0,0,1],
            [1, 0,0,0, 0,0,1,1,0,1],
            [1,0,0,0, 0, 1,0,1,0,1],
            [2,1,0,0,0,1,1,0,1,1],
            [1,1,0,0,0,0,0,1,0,1],
            [3,2,1,1,1,1,1,2,1,0]])


def pmi(M,positive=True):
    """

    :param M:词语共现频次表
    :param positive: 是否采用PPMI的形式
    :return:pmi的值
    """
    col_totals = M.sum(axis=0)
    row_totals = M.sum(axis=1)
    total = col_totals.sum()#总频次
    expected = np.outer(row_totals,col_totals) / total
    M = M / expected

    with np.errstate(divide='ignore'): # 不显示log(0)的警告
        M = np.log(M)

    M[np.isinf(M)] = 0.0 #将log(0)置为0
    if positive:
        M[M<0] = 0.0
    return M


if __name__ == "__main__":
    np.set_printoptions(precision=2)#保留两位小数
    M_pmi = pmi(M)
    print(M_pmi)


    """
    SVD奇异值分解，如果值在奇异值构成的对角矩阵中保存d个最大的奇异值，就被称为阶段奇异值分解，则U的每一行为相应词的d维向量表示
    """
    U,s,Vh = np.linalg.svd(M_pmi)
    words = ["我","喜欢","自然","语言","处理","爱","深度","学习","机器","。"]
    for i in range(len(words)):
        plt.text(U[i,0],U[i,1],words[i])#U中前两维对应二维空间的坐标

    plt.show()
