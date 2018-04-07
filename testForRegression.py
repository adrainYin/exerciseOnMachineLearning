from numpy import *
import regression
import matplotlib.pyplot as plt

dataMat , labelMat = regression.loadData('CH08_data/ex0.txt')
w = regression.standregression(dataMat , labelMat)
print(w)

dataMatrix = mat(dataMat)
labelMatrix = mat(labelMat)
predictedY = dataMatrix * w
#绘制原始的数据点图像
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMatrix[:,1].flatten().A[0] , labelMatrix.T[:,0].flatten().A[0])
#绘制预测函数图像
xMatrix = dataMatrix.copy()
xMatrix.sort(0)
#此时得到的yMatrix是预测值
yMatrix = xMatrix * w
ax.plot(xMatrix[:,1] , yMatrix)
plt.show()
#计算相关系数
corrNum = corrcoef(labelMatrix , predictedY.T)
print('相关系数为 %d' , corrNum)