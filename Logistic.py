from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('CH05_data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat , labelMat

#定义sigmoid函数
def sigmoid(val):
    return 1.0 / (1 + exp(-val))

#定义梯度上升法
def gradAscent(dataMat,labelMat):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).transpose()  #进行矩阵的转置
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMatrix - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

#画出曲线
#mat函数是转换矩阵函数，array是转换numpy数组的函数。getA()是将矩阵转换为nunpy数组的函数 mat和array可以相互转换
def plotBestFit(weigths):
    import matplotlib.pyplot as plt
    dataMat , labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if (int(labelMat[i]) == 1):
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)  #返回一个array数组 其中参数值代表着起始，终止，迭代间隔
    y = (-weigths[0] - weigths[1] * x) / weigths[2]   #返回的是一个数
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度下降法
def dtocGradAscent0(dataMatrix , classLabel):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weigths = ones(n)  #返回的是矩阵吗
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weigths))
        error = classLabel[i] - h
        weigths = weigths + alpha * error * dataMatrix[i]
    return weigths

#加入参数修正的随机梯度上升法
def stocGradscent1(dataMatrix , labelClass , numIter = 150):#迭代次数参数的150是缺省值，如果不输入的话那么默认的迭代次数就是150
    m,n = shape(dataMatrix)
    weigths = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))  #加入随机取样的环节
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01 #alpha是变长属性，并且随着迭代次数的增多，步长会越来越小
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weigths))
            error = labelClass[randIndex]  - h
            weigths = weigths + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weigths

#真是案例，预测马的疝气死亡率
def classifyVector(inX , weigths):
    prob = sigmoid(sum(inX * weigths))
    if prob > 0.5:
        return 1.0
    else:
        return 0.5

