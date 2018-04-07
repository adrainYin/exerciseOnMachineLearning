from numpy import *

def loadData(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        #将当前数据行拆分为数组
        currLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat , labelMat

def standregression(dataMat , labelMat):
    dataMatrix = mat(dataMat)
    labelVector = mat(labelMat).T
    xTx = dataMatrix.T * dataMatrix
    if linalg.det(xTx) == 0.0:
        print('这个矩阵的行列式为 0 ，不能做转置运算')
        return
    w = xTx.I * dataMatrix.T * labelVector
    return w

#计算的是待测点和数据矩阵的高斯距离
def lwlr(testPoint , dataMat , labelMat , k = 1.0):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).T
    m = shape(dataMatrix)[0]
    weights = mat(eye(m))
    #生成测试点对应的高斯矩阵
    for j in range(m):
        dist = testPoint - dataMatrix[:,j]
        weights[j,j] = exp(dist * dist.T / (-2 * k**2))


