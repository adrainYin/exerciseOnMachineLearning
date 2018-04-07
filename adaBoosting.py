from numpy import *

def loadSimpleData():
    dataMat = matrix([[1. , 2.1],
                      [2. , 1.1],
                      [1.3 , 1.],
                      [1. , 1.],
                      [2. , 1.]])
    classLabel = [1.0 , 1.0 , -1.0 , -1.0 , 1.0]
    return dataMat , classLabel

#简单分类函数，通过对阈值判断执行分类结果
"""dataMatrix是数据矩阵 ， dimen是分类的维数 ， threshVal是阈值 ， threshIneq是分类需要的类别信息
"""
def stumpClassify(dataMatrix , dimen , threshVal , threshIneq):
    retArray = ones((shape(dataMatrix)[0] ,1))
    if threshIneq == 'lt':
        #在这里进行了一个判断 ， 如果dime维的数据在左边，那么这些数据的分类全为-1 因为初始化的时候
        #分类已经设置为了+1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0

    return retArray

"""基本的单层决策树算法，训练出的为一个弱分类器
"""
def buildStump(dataArr , classLabels , D):
    dataMatrix = mat(dataArr)
    classMatrix = mat(classLabels).T
    m , n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m , 1)))
    minError = inf
    for i in range(n):
        minRange = dataMatrix[:,i].min()
        maxRange = dataMatrix[:,i].max()
        #确定阈值修改的步长
        stepSize = (maxRange - minRange) / numSteps
        #这里定义j从-1开始就意味着阈值的选择可以超过数据的范围 ，即在调用stumpClassify函数时可以将全部数据分成一类
        for j in range(-1 , int(numSteps) + 1):
            for inequal in ['lt' , 'gt']:
                threshVal = minRange + float(j) * stepSize
                #此时的分类结果有1 和-1之分
                predictArray = stumpClassify(dataMatrix , i , threshVal , inequal)
                errArray = mat(ones((m,1)))
                #这样处理的目的是方便之后的权值错误率计算
                errArray[predictArray == classMatrix] = 0
                #得到权值后的错误率，如果分类正确，那么在内积中的相应项数为0
                weightError = D.T * errArray
                print('split: dim %d , thresh %.2f , thresh inequal: %s , weigthError : %.3f' % (i,threshVal,inequal,weightError))
                #如果取得的分类效果要更好，那么就保存该次的分类信息
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictArray.copy()
                    bestStump['dimen'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['threshIneq'] = inequal
    return bestStump , minError , bestClassEst

"""使用adaBoost算法逐渐提升弱分类器的分类能力"""
"""numIt是迭代获得的弱分类数目 ， 算法停止的条件有：(1)分类错误率为0  (2)达到最大的弱分类器数目值"""
def adaBoostingTrain(dataArr , classLabels , numIt = 40):
    #数据的初始化
    weakClassArr = []
    m , n = shape(dataArr)
    D = mat(ones((m ,1)) / m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump , error , classEst = buildStump(dataArr , classLabels , D)
        # D.T
        alpha = float(0.5 * log((1.0 - error) / max(error , 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #转置成行向量
        # classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print('aggClassEst : ' , aggClassEst.T)
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T , ones((m,1)))
        erroeRate = aggError.sum() / m
        print('total error : ', erroeRate ,'\n')
        if erroeRate == 0.0:
            break
    return weakClassArr , aggClassEst


"""这里定义的函数是用多分类器对数据进行分类。其中classifierArr是多分类器数组。训练好的分类器存入,datToClass是测试样本
    返回值是数据的预测分类"""
def adaClassify(datToClass , classsifierArr):
    dataMatrix = mat(datToClass)
    m , n = shape(dataMatrix)
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classsifierArr)):
        classEst = stumpClassify(dataMatrix , classsifierArr[i]['dimen'] , classsifierArr[i]['threshVal'] ,
                                 classsifierArr[i]['threshIneq'])
        aggClassEst += classsifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


def loadDataSet(filename):
    #取得列表的列个数
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        #将数据内容加入列表
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat , labelMat

"""绘制ROC曲线函数 ， 有两个参数
第一个参数是分类器的预测强度，也就是没有经过标准化吃处理的原始的分类数据。输入格式是向量或者numpy矩阵
第二个参数是数据的真实分类情况
"""
def plotROC(predStrengths , classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0 ,1.0)
    ySum = 0.0
    #取得真实分类为正例的数量
    numPosClass = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClass)
    xStep = 1 / float(len(classLabels) - numPosClass)
    #对预测强度按照从强到弱的顺序进行排序
    sortedPredStrengths = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedPredStrengths.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0.0
            delY = yStep
        else:
            delX = xStep
            delY = 0.0
            ySum += cur[1]
        ax.plot([cur[0] , cur[0] - delX] , [cur[1] , cur[1] - delY] , c = 'b')
        cur = (cur[0] - delX , cur[1] - delY)
    #绘制plot图形， 第一个参数是x轴的点坐标，第二个参数是y轴的点坐标，第三个参数是对颜色的设置
    ax.plot([0,1] , [0,1] ,'b--')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positivee Rate')
    plt.title('ROC curve')
    ax.axis([0,1,0,1])
    plt.show()
    print('the Area Under the curve is :' , ySum * xStep)
