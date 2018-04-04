from numpy import *

#该函数的作用是从文件中读取指定的数据，并转换为数据集合和标签集合
def loadDataSet (filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for lines in fr.readlines():
        #按行读取数据，并且按照空格划分数据
        lineArr = lines.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        #zheli必须要加入float,否则得到的数据都是字符类型，必须进行强制转换
        labelMat.append(float(lineArr[2]))
    return dataMat , labelMat

#函数功能：当输入的值只要不等于i ， 那么j就会随机选择。感觉是一个死循环
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

#根据H和L的值调整aj的值，使得aj一直处于H和L的中间
#因为根据约束条件可得aiyi + ajyj = C是一个固定的值，那么再根据KKT的约束条件，ai > = 0
#又因为再SVM汇总，y值的选择只有两种，不是1 就是 -1 ， 那么会对ai和aj的选择造成限制， 那么就需要实时调整aj的值
def clipAlpha(aj , H , L):
    if(aj > H):
        aj = H
    if(aj < L):
        aj = L
    return aj

#定义输入参数为：数据集，类别标签，常数C ,容错率 和最大迭代次数。在这里常数C就是对参数调整的限制
def smosimple(dataMatIN , classLabels , C , toler , maxIter):
    dataMatrix = mat(dataMatIN)
    labelMat = mat(classLabels).transpose()
#初始化所有的变量
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
  #正式开始循环
    while(iter < maxIter):
        alphaPairsChange = 0
        for i in range(m):
            #相乘操作和.T的转置操作，alphas和labelMat都是列向量，用nultiply为点积相乘，相乘之后还为列向量
            #这一步是计算预测值，因为参数矩阵已经可以被表示出来
            fxi = float((multiply(alphas,labelMat).T) * (dataMatrix * dataMatrix[i,:].T)) + b
            #计算误差值
            Ei = fxi - float(labelMat[i])
            #这里的toler就是软间隔的阈值，如果满足条件则进入循环调整。
            #这里取alphas[i]是因为如果参数需要调整，那么参与迭代的点一定是支持向量上的点或者软间隔点，因为支持向量外的点一定有alphas[i]=0
            #此处的if语句为判断当前alpha参数是否需要加入调整。如果大于阈值那么就加入调整，作为选择的第一个参数
            if(((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0))):
                j = selectJrand(i,m)
                fxj = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                #计算j数据的误差
                Ej = fxj - float(labelMat[i])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0 , alphas[j] - alphas[i])
                    H = min(C , C + alphas[j] - alphas[i])
                else:
                    L = max(0 , alphas[j] + alphas[i] -C)
                    H = min(C , alphas[i] + alphas[j])
            if (L == H): print('L == H'); continue
            #对alphaJ进行更新
            #计算的是eta的值，是两个两个向量的内积，2倍的x1与x2的内积减去x1的内积再减去x2的内积 ，为最优化a2的分母
            eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T -\
                dataMatrix[j,:] * dataMatrix[j,:].T
            #eta = 2.0 * alphas[i] * alphas[i].T - alphas[i] * alphas[j].T - alphas[j] * alphas[j].T
            #判断表示该参数值，为什么不能大于0？
            if eta >= 0.0: print('eta >=0.0'); continue
            #根据公式更新后的aj的值
            alphas[j] = alphas[j] - (labelMat[j] * (Ei - Ej) / eta)
            #根据KKT约束对参数进行调整
            alphas[j] = clipAlpha(alphas[j] , L , H)
            if((abs(alphas[j]) - alphaJold) < 0.00001):
                print('j not moving enough')
                continue
            #计算得到i的更新公式
            alphas[i] = alphas[i] + labelMat[i] * labelMat[j] * (alphaJold - alphas[j])
            #更新线性系数b
            b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
            dataMatrix[i,:] * dataMatrix[i,:].T - \
            labelMat[j] * (alphas[j] - alphaJold) * \
            dataMatrix[i,:] * dataMatrix[j,:].T

            b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
            dataMatrix[i,:] * dataMatrix[j,:].T - \
            labelMat[j] * (alphas[j] - alphaJold) * \
            dataMatrix[j,:] * dataMatrix[j,:].T
            #判断alpha值是否存在KKT约束中，然后更新b的值
            if((0 < alphas[i]) and (C > alphas[i])): b = b1
            elif((0 < alphas[j]) and (C > alphas[j])): b = b2
            else: b = (b1 + b2) / 2.0
            #标记位，表明有参数经过改动，所以迭代次数要重置
            alphaPairsChange += 1
            print('iter : %d , i %d , pairs changed %d', iter , i , alphaPairsChange)
        if(alphaPairsChange == 0): iter += 1
        else:iter = 0
        print('iteration number : %d' , iter)
    return b , alphas



#完整的smo优化算法，更改了选择参数的方式：
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,totor):
        #定义了数据剧矩阵
        self.X = dataMatIn
        #定义了标签矩阵
        self.labelMat = classLabels
        self.C = C
        self.totor = totor
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        #定义EI的缓存数组，为m * 2的形式
        self.eCache = mat(zeros((self.m,2)))

#函数的作用：传入自定义的最优化结构和待选的数据k，然后计算误差并返回
# def calcEK(OS, k):
#     FXK = float(multiply(OS.alphas , OS.labelMat).T * (OS.X * OS.X[k,:].T)) + OS.b
#     EK = FXK - float(OS.labelMat[k])
#     return EK

def calcEK(OS , k):
    FXK = float(multiply(OS.alphas , OS.labelMat).T * OS.K[:,k] + OS.b)
    EK = FXK - float(OS.labelMat[k])
    return EK

def selectJ(i,OS,EI):
    maxK = -1;maxDeltaE = 0;EJ = 0
    OS.eCache[i] = [1,EI]
    #选出非零的E值所对应的下标。因为在类初始化的时候 ，所有的E均倍设置为了零值
    #存储值，保存暂时的数据
    validECacheList = nonzero(OS.eCache[:,0].A)[0]
    if(len(validECacheList) > 1):
        for k in validECacheList:
            if k ==i:continue
            EK = calcEK(OS,k)
            deltaE = abs(EI - EK)
            if(deltaE > maxDeltaE):
                maxDeltaE = deltaE
                maxK = k
                EJ = EK
        return maxK , EJ
    else:
        j = selectJrand(i,OS.m)
        EJ = calcEK(OS,j)
        return j,EJ

def update(OS , k):
    EK = calcEK(OS,k)
    OS.eCache[k] = [1,EK]

#该函数是完整的pallt smo算法的选择部分
def innerL(i, OS):
    Ei = calcEK(OS,i)
    #选择的i值满足优化条件
    if(((OS.labelMat[i] * Ei < -OS.totor) and (OS.alphas[i] < OS.C)) or
            ((OS.labelMat[i] * Ei > OS.totor) and (OS.alphas[i] > 0))):
        j,Ej,= selectJ(i , OS , Ei)
        alphaIold = OS.alphas[i].copy()
        alphaJold = OS.alphas[j].copy()
        if(OS.labelMat[i] != OS.labelMat[j]):
            L = max(0 , OS.alphas[j] - OS.alphas[i])
            H = min(OS.C , OS.C + OS.alphas[j] - OS.alphas[i])
        else:
            L = max(0 , OS.alphas[j] + OS.alphas[i] - OS.C)
            H = min(OS.C , OS.alphas[j] + OS.alphas[i])
        if(L == H):print('L == H'); return 0
        #eta = 2.0 * OS.X[i,:] * OS.X[j,:].T - OS.X[i,:] * OS.X[i,:].T - OS.X[j,:] * OS.X[j,:].T
        eta = 2.0 * OS.K[i,j] - OS.K[i,i] - OS.K[j,j]
        if(eta >= 0):print('eta >=0');return 0
        OS.alphas[j] -= OS.labelMat[j] * (Ei - Ej) / eta
        OS.alphas[j] = clipAlpha(OS.alphas[j] , H , L)
        #保存更新后的EK误差值
        update(OS,j)
        if(abs(OS.alphas[j] - alphaJold) < 0.0001):
            print('J not moving enough');return 0
        OS.alphas[i] = alphaIold + OS.labelMat[i] * OS.labelMat[j] * (alphaJold - OS.alphas[j])
        update(OS,i)
        #计算两个b的值
        # b1 = OS.b - Ei - OS.labelMat[i]* (OS.alphas[i] - alphaIold) *\
        # OS.X[i,:] * OS.X[i,:].T - OS.labelMat[j] *\
        #      (OS.alphas[j] - alphaJold) * OS.X[i,:] * OS.X[j,:].T

        b1 = OS.b - Ei - OS.labelMat[i] * (OS.alphas[i] - alphaIold) * OS.K[i,i] - \
            OS.labelMat[j] * (OS.alphas[j] - alphaJold) * OS.K[i,j]
        # b2 = OS.b - Ej - OS.labelMat[i] * (OS.alphas[i] - alphaIold) *\
        # OS.X[i,:] * OS.X[j,:].T - OS.labelMat[j] *\
        #      (OS.alphas[j] - alphaJold) * OS.X[j,:] * OS.X[j,:].T
        b2 = OS.b - Ej - OS.labelMat[i] * (OS.alphas[i] - alphaIold) * OS.K[i,j] -\
            OS.labelMat[j] * (OS.alphas[j] - alphaJold) * OS.K[j,j]
        if(OS.alphas[i] > 0) and (OS.alphas[i] < OS.C): OS.b = b1
        elif(OS.alphas[j] > 0) and (OS.alphas[j] < OS.C): OS.b = b2
        else:OS.b = (b1 + b2) / 2.0
        return 1
    #如果不满足，则直接跳出if语句，再次选择i值
    else:return 0

#外循环选择函数
def smoP(dataMaIn , labelMat , C , totor , maxIter , kTup = ('lin',0)):
    OS = newOptStruct(mat(dataMaIn) , mat(labelMat).transpose() , C , totor , kTup)
    iter = 0
    entirSet = True
    alphaPairsChanged = 0
    #主循环函数
    while((iter < maxIter) and ((entirSet) or (alphaPairsChanged > 0))):
        alphaPairsChanged = 0
        if entirSet:
            for i in range(OS.m):
                alphaPairsChanged += innerL(i,OS)
                print('fullset , iter: %d , i:%d , pairs changed %d',(iter , i , alphaPairsChanged))
            iter += 1
        else:
            #什么的意思
            nonboudIs = nonzero((OS.alphas.A > 0 ) * (OS.alphas.A < C))[0]
            for i in nonboudIs:
                alphaPairsChanged += innerL(i,OS)
                print('fullset , iter: %d , i:%d , pairs changed %d', (iter, i, alphaPairsChanged))
            iter += 1
        if(entirSet):entirSet = False
        elif(alphaPairsChanged == 0 ):entirSet = True
        print('iternumber = %d', iter)
    return OS.b , OS.alphas


#定义带有核函数的支持向量机,返回K(x,y)的向量
#这里的X是整个数据矩阵，A是待计算的数据
def kernelTrans(X , A , KTup):
    m , n = shape(X)
    K = mat(zeros((m , 1)))
    if(KTup[0] == 'lin'):
        K = X * A.T
    elif(KTup[0] == 'rbf'):
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * KTup[1]**2))
    else:
        raise NameError('Kernal is not recognized!')
    return K

class newOptStruct:
    def __init__(self,dataMatIn , classLabel , C , totor , kTup):#这里的kTup是选择的核函数的类型
        self.X = dataMatIn
        self.labelMat = classLabel
        self.C = C
        self.totor = totor
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X , self.X[i,:] , kTup)


def testRbf(k1 = 1.3):
    dataArr , labelArr = loadDataSet('CH06_data/testSetrBF.txt')
    b , alphas = smoP(dataArr , labelArr , 200 , 0.0001 , 10000 , ('rbf' , k1))
    # print(b)
    # print(alphas)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    #取出非零的alpha值，代表的是全部的支持向量点
    svInd = nonzero(alphas > 0)[0]
    sVs = dataMat[svInd]
    print('***********************')
    print(sVs)
    print('***********************')
    labelSv = labelMat[svInd]
    #print('there are $d support vectors' % shape(sVs))
    m,n = shape(dataMat)
    errorCount = 0.0
    #定义如何对数据进行分类，因为已经训练出了支持向量，所以在预测的过程中只用使用支持向量点
    for i in range(m):
        kernalEval = kernelTrans(sVs , dataMat[i,:],('rbf',k1))
        predict = kernalEval.T * multiply(labelSv , alphas[svInd]) + b
        if(sign(predict) != sign(labelArr[i])):
            errorCount += 1
    print('分类的错误率是 %f'  % (float(errorCount / m)))
    print('\n')
    print('*******************我是华丽的分割线*******************')
    dataArr, labelArr = loadDataSet('CH06_data/testSetrBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    # print(b)
    # print(alphas)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    # 取出非零的alpha值，代表的是全部的支持向量点
    svInd = nonzero(alphas > 0)[0]
    sVs = dataMat[svInd]
    print('***********************')
    print(sVs)
    print('***********************')
    labelSv = labelMat[svInd]
    # print('there are $d support vectors' % shape(sVs))
    m, n = shape(dataMat)
    errorCount = 0.0
    # 定义如何对数据进行分类，因为已经训练出了支持向量，所以在预测的过程中只用使用支持向量点
    for i in range(m):
        kernalEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernalEval.T * multiply(labelSv, alphas[svInd]) + b
        if (sign(predict) != sign(labelArr[i])):
            errorCount += 1
    print('分类的错误率是 %f' % (float(errorCount / m)))
