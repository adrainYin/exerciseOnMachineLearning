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
