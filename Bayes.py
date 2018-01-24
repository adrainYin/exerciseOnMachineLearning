from numpy import *
import random


def loadData2Vertex():
    postingList = [['my','dog','has','fea'],['maybe','not','take','him','to','dog','park'],
                   ['Can','you','help','me'],['stop','posting','stupid','worthless','garbage'],
                   ['how','to','stop','him'],['quite','buying','worthless','dog','food']]

    labelVertex = [0,1,0,1,0,1]
    return postingList,labelVertex

def createZVocaList(dataSet):  #dataSet : metrix
    #vocaSet = set([])  #初始化set的集合
    vocaSet = set()  #初始化Set集合
    for documents in dataSet:
        vocaSet = vocaSet | set(documents)
    return list(vocaSet)  #返回的vocaList是list类型


#对Set集合内部的元素计数
def setWords2Vec(vocaList,inputSet):
    retuenVec = [0] * len(vocaList)
    for words in inputSet:
        if words in vocaList:
            retuenVec[vocaList.index(words)] += 1  #list列表的index方法，返回该元素在list中的下标，否则抛出空指针异常
        else:
            print("this word in not in the vocaList")
    return retuenVec #返回单词向量的单词存在表

def trainingBayes(trainMartix,trainCategory):
    numTrainDocus = len(trainMartix)
    numWords = len(trainMartix[0]) #文档一个段内的单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocus) #训练集中辱骂的比例
    p0num = ones(numWords)  #防止有0项出现，使得乘积最终为0，所以要用到拉普拉斯正则修正
    p1num = ones(numWords)
    p0denom = p1denom = 2.0
    for i in range(numTrainDocus):
        if trainCategory[i] == 1:
            p1num += trainMartix[i] #行向量的相加
            p1denom += sum(trainMartix[i]) #计算总数,单词出现的总数
        else:
            p0num += trainMartix[i]
            p1denom += sum(trainMartix[i])

    p1Vec = log(p1num / p1denom)
    p0Vec = log(p0num / p0denom)
    return p1Vec,p0Vec,pAbusive  #返回值也是向量，返回的是概率向量

def bayesCLassify(vec2CLassify,p1vec,p0vec,pClass):
    p1 = sum(vec2CLassify * p1vec) + log(pClass)  #分类向量与概率向量的相乘
    p0 = sum(vec2CLassify * p0vec) + log(1.0 - pClass)
    if p1 > p0:
        return 1
    else:
        return 0

def testingBayes(): #预测函数
    postingList , labelVertex = loadData2Vertex()
    vocaSet = createZVocaList(postingList)
    trainMat = []
    for postInDocus in postingList:
        trainMat.append(setWords2Vec(vocaSet,postInDocus))

    #array转换为numpy数组
    p1Vec,p0Vec,pAb = trainingBayes(array(trainMat),array(labelVertex))
    testEntry = ['dog','to','me']  #测试用例
    testVec = setWords2Vec(vocaSet,testEntry)
    print("classify as "  , bayesCLassify(testVec,p0Vec,p1Vec,pAb))

#用正则表达式切分字符串
def textParse(string):
    import re
    stringList = re.split(r'\W*',string)
    return [tok.lower() for tok in stringList if len(tok) > 2]


def spamText():
    docList = []
    classList = []
    fullText = []
    for i in range(25):
        wordText = textParse(open('CH04_data/spam/%d.txt' % (i+1)).read()) #读文件，使用通配符
        docList.append(wordText)
        fullText.extend(wordText)
        classList.append(1) #默认的是而分类的类型
        wordText = textParse(open('CH04_data/ham/%d.txt' % (i+1)).read())
        docList.append(wordText)
        fullText.extend(wordText)
        classList.append(0)
        vocList = createZVocaList(docList)

    #选择交叉验证集
    trainingSet = list(range(50)) #python3.x的改动，冉哥返回的是range对象，需要数组则需要加list()
    testSet = []  #返回的是所有的测试集的序号
    for i in range(10):
        index = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[index])
        del(trainingSet[index])
    trainMat = []  #所有的训练数据
    trainCLass = [] #所有的训练的标签
    for docIndex in testSet:
        trainMat.append(setWords2Vec(vocList,docList[docIndex]))
        trainCLass.append(classList[docIndex])
    p1Vec,p0Vec,pAb = trainingBayes(array(trainMat),array(trainCLass))  #转换为numpy中的数组类型
    errorCount = 0.0
    for testIndex in testSet:
        testVec = setWords2Vec(vocList,docList[testIndex])
        if (bayesCLassify(testVec,p1Vec,p0Vec,pAb) != classList[testIndex]):
            errorCount += 1
    #print('错误率是 ', float(errorCount / len(classList)))
    return float(errorCount / len(classList))
