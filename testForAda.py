import adaBoosting
from numpy import *

# dataMat , labelMat = adaBoosting.loadSimpleData()
# # D = mat(ones((5,1))/5)
# # bestStump , minError , bestClassEst = adaBoosting.buildStump(dataMat , labelMat , D)
# # print(bestStump)
# # print(minError)
# # print(bestClassEst)
# weakClassArr = adaBoosting.adaBoostingTrain(dataMat , mat(labelMat) , 9)
# print(weakClassArr)
# print('******************我是华丽的分割线*****************')
# comeout = adaBoosting.adaClassify([0,0] , weakClassArr)
# print(comeout)

dataMat , labelMat = adaBoosting.loadDataSet('CH07_data/horseColicTraining2.txt')
weakClassArr = adaBoosting.adaBoostingTrain(dataMat , labelMat , 500)
testArr , labelArr = adaBoosting.loadDataSet('CH07_data/horseColicTest2.txt')
m ,n = shape(testArr)
#获得所有的预测分类信息
predictions = adaBoosting.adaClassify(testArr , weakClassArr)
#计算分类错误率
errArr = mat(ones((m,1)))
errnum =  errArr[predictions != mat(labelArr).T].sum()
print(errnum)
print('错误率为： %f'  % float(errnum / m))

