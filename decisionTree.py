from math import log
import operator
import pickle
import json

"""信息熵的计算
"""
def computeEnt(dataSet):
    m = len(dataSet)
    labelClass= {}  #使用字典统计不同划分的个数
    for featVec in dataSet:
        currFeat = featVec[-1]  #-1是list集合的最右边的元素，代表着样本的分类
        if(currFeat not in labelClass.keys()):
            labelClass[currFeat] = 0
        labelClass[currFeat] +=1
    shannoEnt = 0.0
    for key in labelClass:
        currFeatNum = labelClass[key]
        prob = float(currFeatNum / m)
        shannoEnt -= prob * log(prob,2)
    return shannoEnt

def creatData():
    dataSet = [[1,1,1],
               [1,1,1],
               [1,0,0],
               [0,1,0],
               [0,1,0],
               [0,0,1]]
    label = ['no surfacing','flippers']
    return dataSet,label


"""
@从文件读数据，转为list集合
"""
def file2list(filenmae):
    fr = open(filenmae)
    lines = fr.readlines()  #返回的是集合
    linesRow = len(lines)
    dataSet = []
    for currLine in lines:
        currLine = currLine.strip()
        dataList = currLine.split()
        dataSet.append(dataList)
    return dataSet

# axis是列号索引，value是具体的值
def splitClass(dataSet,axis,value):
    resultSet = []
    for item in dataSet:
        if(item[axis] == value):
            #对属性值进行抽取，再拼接成list
            extendVec = item[:axis]
            extendVec.extend(item[axis+1 :])
            resultSet.append(extendVec)
    return resultSet

"""我的方法错误，因为会出现log上为0的情况。不应该这样的计算"""
def computeEntOnFeat(dataSet,axis,label): #label属性是分类的取值范围，定义为离散的有限集合list
    m = len(dataSet)
    n = len(label)
    featClass = {}
    featList = [] #定义分类向量，
    for dataItem in dataSet:
        className = dataItem[axis]
        classLabel = dataItem[-1]
        labelIndex = label.index(classLabel)
        if (className not in featClass.keys()):  #统计样本中某属性的取值范围和对应的样本数量
            featClass[className] = 0
        featClass[className] += 1
        flag = 0
        for i in range(len(featList)):
            if (className != featList[i][0]):  #遍历className
                continue
            flag = 1
            featList[i][1] += 1
            featList[i][2+labelIndex] += 1
        if (flag == 0):  #不存在
            classList = [className,1]
            initList = []   #初始化一个长度为n的全0的数组
            for i in range(n):
                initList.append(0)
            classList.extend(initList) #初始化属性集合
            classList[2+labelIndex] +=1
            featList.append(classList)
    classNum = len(featList)  #所有属性的取值的范围
    shannoEnt = computeEnt(dataSet)
    for feature in featList:  #feature是每一个属性可能取值
        numCount = feature[1]
        predClass = float(numCount / m)  #某个属性取值的比例
        shannoEntForFeature = 0.0
        for i in range(len(label)):
            pred = float(feature[2+i] / numCount)
            shannoEntForFeature -= pred * log(pred,2)  #某个属性的取值的香农值
        shannoEnt -= float(predClass * shannoEntForFeature)
    return shannoEnt,featList

def chooseBestFeature(dataSet):
    n = len(dataSet)
    numFeature = len(dataSet[0]) -1
    baseEnt = computeEnt(dataSet)
    bestEnt = 0.0  #初始化信息增益最小值
    bestFeature = -1 #用列号表示
    for i in range(numFeature):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)  #统计出某属性取值的范围
        newEnt = 0.0
        for value in uniqueVals:
            resultData = splitClass(dataSet,i,value)  #分割数据为特定属性下的特定取值
            m = len(resultData)
            prodValue = float(m / n)
            prod = computeEnt(resultData)
            newEnt += prodValue * prod
        gainEnt = baseEnt - newEnt
        if (gainEnt > bestEnt):
            bestEnt = gainEnt
            bestFeature = i
    return bestFeature

"""返回最多的分类标签"""
def majorityClass(classList):
    classLabel = {}
    for label in classList:
        if(label not in classLabel.keys()):
            classLabel[label] = 0
        classLabel[label] += 1
    sortClassLabel = sorted(classLabel.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassLabel[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #分类标签的属性完全相同
        return classList[0]
    if len(dataSet[0]) == 1: #分类属性全部使用，仍没有生成叶子结点
        return majorityClass(classList)
    #进行递归调用
    bestFeature = chooseBestFeature(dataSet)
    bestLabel = labels[bestFeature]  #返回特征的标签
    #定义的嵌套的字典
    """
    如果递归结束则返回最终的label
    如果不是则返回字典，继续递归
    """
    tree = {bestLabel:{}}  #重新生成嵌套循环
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        resultSet = splitClass(dataSet,bestFeature,value)
        subLabels = labels[:]
        tree[bestLabel][value] = createTree(resultSet,subLabels)
    return tree

def storeTree(tree,filename):
    # fw = open(filename,'w')
    # pickle.dump(tree,fw,)
    # fw.close()
    fw = open(filename,'w')
    json.dump(tree,fw)

def getTree(filename):
    fr = open(filename)
    # return pickle.load(fr)
    return json.load(fr)