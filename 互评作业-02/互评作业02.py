import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


#数据导入与处理
def loadDataSet():
    #导入数据并显示
    winedata = pd.read_csv('D:/Python/jupyter-notebook/data/winemag-data_first150k.csv')
    winedata = winedata.dropna()
    winedata.head(n=100)  # 显示前100行

    winedata = winedata.drop(columns=['Unnamed: 0', 'description'])

    winedata.info()

    winedata['price'].loc[winedata['price'] < 200].hist(bins=20)
    plt.title("price")
    plt.figure()
    winedata['points'].hist()
    plt.title("points")

    # 进行离散化分段处理：cut函数
    bin = [0, 20, 30, 40, 50, 60, 200]
    winedata['price'] = pd.cut(winedata['price'], bin)
    winedata['price'] = winedata['price'].astype('str')
    # 进行离散化分段处理：cut函数
    bin = [0, 75, 80, 85, 90, 95, 100]
    winedata['points'] = pd.cut(winedata['points'], bin)
    winedata['points'] = winedata['points'].astype('str')


def createC1(dataSet):
    """
    构建初始候选项集的列表，即所有候选项集只包含一个元素，
    C1是大小为1的所有候选项集的集合
    """
    C1 = []
    for transaction in np.array(dataSet):
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """
    计算Ck中的项集在数据集合D(记录或者transactions)中的支持度,
    返回满足最小支持度的项集的集合，和所有项集支持度信息的字典。
    """
    ssCnt = {}
    for tid in D:
        # 对于每一条transaction
        if Ck is not None:
            for can in Ck:
                # 对于每一个候选项集can，检查是否是transaction的一部分
                # 即该候选can是否得到transaction的支持
                if can.issubset(tid):
                    ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 每个项集的支持度
        support = ssCnt[key] / numItems

        # 将满足最小支持度的项集，加入retList
        if support >= minSupport:
            retList.insert(0, key)

            # 汇总支持度数据
            supportData[key] = support
    return retList, supportData


# Aprior算法
def aprioriGen(Lk, k):  # create ck(k项集)
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()  # 排序
            if L1 == L2:  # 比较i,j前k-1个项若相同，和合并它俩
                retList.append(Lk[i] | Lk[j])  # 加入新的k项集 | stanf for union
    return retList # ck

def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet) # c1 = return map
    # D = map(set, dataSet) # D = map
    D = dataSet
    L1, supportData = scanD(D, C1, minSupport)  # 利用k项集生成频繁k项集（即满足最小支持率的k项集）
    L = [L1]  # L保存所有频繁项集

    k = 2
    while (len(L[k - 2]) > 0):  # 直到频繁k-1项集为空
        Ck = aprioriGen(L[k - 2], k)  # 利用频繁k-1项集 生成k项集
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)  # 保存新的频繁项集与其支持度
        L.append(Lk)  # 保存频繁k项集
        k += 1
    return L, supportData  # 返回所有频繁项集，与其相应的支持率


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    lift = []
    file = open("generate_rules.txt","a",encoding = "utf-8")
    for conseq in H:  # 后件中的每个元素
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            file.write(str(freqSet - conseq)+"-->"+str(conseq)+" support:"+str(supportData[freqSet])+" conf:"+str(conf)+'\n')
            brl.append((freqSet - conseq, conseq, supportData[freqSet], conf))  # 添加入规则集中
            prunedH.append(conseq)  # 添加入被修剪过的H中
    file.close()
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])  # H是一系列后件长度相同的规则，所以取H0的长度即可
    if (len(freqSet) > m + 1):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []  # 存储规则
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def lift_eval(rules, suppData): # lift evaluation
    # lift(A, B) = P(A交B) / (P(A) * P(B)) = P(A) * P(B | A) / (P(A) * P(B)) = P(B | A) / P(B) = confidence(A— > B) / support(B) = confidence(B— > A) / support(A)
    lift = []
    for rule in rules:
        freqSet_conseq = rule[0]
        conseq = rule[1]
        lift_val = float(rule[3]) / float(suppData[rule[1]])
        lift.append([freqSet_conseq,conseq,lift_val])
    return lift

dataSet = loadDataSet()
L, suppData = apriori(dataSet)
print(L)
rules = generateRules(L, suppData, minConf=0.5)
print(rules)
lifts = lift_eval(rules, suppData)
print(lifts)