from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

def creatDataset():

    # group = array([[1,1],[1,1],[0,0],[0,0.1]])
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group ,labels


def classify0(intX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]
    # diffMat = tile(intX ,(dataSetSize,1)) - dataSet
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet # Subtract element-wise

    # tile重复代码行，将array 重读为一个二维的情况，行重复dataSet的大小，列重复为1
    sqDiffMAt = diffMat **2
    sqDistance = sqDiffMAt.sum(axis=1)
    # axis = 1 表示第二维度，设定维度从0开始
    distance = sqDistance **0.5
    sortdDistanceIndices = distance.argsort()
    #argsort函数返回一个数组，是从小到大顺序排列的序列号，构成一个array
    classCount = {}
    for i in range(k):
        voteLable = labels[sortdDistanceIndices[i]]
        classCount[voteLable] = classCount.get(voteLable,0)+1  #用字典的形式实现，key是label标签，value是出现的次数
        sortClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)

    return sortClassCount[0][0]




def file2matrix(filename):
    fr = open(filename) #读文件,需要在当前目录下加一个文件夹名称，而且是双引号"CH02_data/datingTestSet.txt"
    arrayfLines = fr.readlines()  #返回的是一个列表，存放每一行的数据
    numOfLines = len(arrayfLines)
    data_matrix = zeros((numOfLines,3)) #类似于matlab初始化矩阵
    label_matrix = []
    index = 0
    for line in arrayfLines:
        line = line.strip()
        string_list = line.split('\t') #数据用tab字符的转义字符分割，一个tab默认是两个字符
        data_matrix[index,:] = string_list[0:3]  #数组下标从0开始，直到第3个，并且不包括第三个
        label_matrix.append(string_list[3])  #只能用list,因为
        index += 1
    return data_matrix,label_matrix

def matPlot(data,label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,1],data[:,2])
    plt.show()

#定义转移函数，将label转移成int类型用矩阵存储
#numpy中array的定义是多维数组，本质上还是数组，所以起始下标还是从0开始
def label2array(list):
    length = len(list)
    labelArray = zeros((length,1))
    temp = []
    for i in range(length):
        label = list[i]
        if label not in temp:
            temp.append(label)
        index = temp.index(label) +1
        labelArray[i] = index
    return labelArray

def autoNorm(dataSet):
    min = dataSet.min(0)  #取最小值，返回值是一个向量
    max = dataSet.max(0)
    range = max - min
    normalData = zeros(shape(dataSet))
    m = dataSet.shape[0]  #取的是dataSet的行数
    normalData = dataSet - tile(min,(m,1))  #tile函数的用法
    normalData = normalData  / tile(max,(m,1))
    return normalData,range,min

def classifyTest():
    hoRatio = 0.1
    data,label = file2matrix("Ch02_data/datingTestSet.txt")
    normalData,ranges,min = autoNorm(data)  #要注意变量名不能和关键字名和函数名重复
    m = normalData.shape[0]
    testDataIndex = int(m*hoRatio)
    error = 0.0
    for i in range(testDataIndex):  #range函数的起始下标也是从0开始
        result = classify0(normalData[i,:],normalData[testDataIndex:m,:],label[testDataIndex:m],3)
        print("分类标签是 %s , 原始标签是 %s" % (result,label[i]))
        if (result != label[i]):
            error +=1.0
    print("错误率是 %f" % (error / float(testDataIndex)))
    print(error)

def predicteClassify():
    labelList = ['notlike','somedoses','largedoses']
    gameTime = float(input("the percentage of time on game"))
    fMiles = float(input("the fly times per year"))
    iceCreame = float(input("the icecream per year"))
    inArray = array([fMiles,gameTime,iceCreame])
    training_data , training_labels = file2matrix("Ch02_data/datingTestSet2.txt")
    normalData,ranges,minValues = autoNorm(training_data)
    normalTest = (inArray - minValues) / ranges
    result = int(classify0(normalTest,normalData,training_labels,3))
    print("对其的喜爱程度是 " ,labelList[result - 1])

def img2vector(filename):
    num_vector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        arrayLine = fr.readline()
        for j in range(32):
            num_vector[0,i*32+j] = arrayLine[j]
    return num_vector

def handwritingTest():
    hwlabels = []
    fileList = listdir("digits/trainingDigits")  #返回文件列表，其中是所有打开文件的集合
    m = len(fileList)
    trainingData = zeros((m,1024))
    #从文件构造训练数据和对应的标签
    for i in range(m):
        #分类手写识别的数字标识
        filename = fileList[i]
        string = filename.split('.')[0]
        number = string.split('_')[0]
        hwlabels.append(number)  #返回的是一个m长度的list，其中元素是所有的数字的集合，定义为标签
        trainingData[i,:] = img2vector("digits/trainingDigits/%s" % filename)

    testFileList = listdir("digits/testDigits")
    errorCount = 0.0
    n = len(testFileList)
    for j in range(n):
        testFile = testFileList[j]
        testFileName = testFile.split('.')[0]
        testLabel = testFileName.split('_')[0]
        testVector = img2vector("digits/testDigits/%s" % testFile)
        result = classify0(testVector,trainingData,hwlabels,3)
        print("识别数字是%s,原始的数字是%s" % (result,testLabel))
        if(result != testLabel):
            errorCount += 1.0
    print("识别错误个数是%d,错误率是%f" % (int(errorCount),errorCount / float(n)))


