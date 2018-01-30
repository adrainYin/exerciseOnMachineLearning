import Logistic
from numpy import *
dataSet , labelSet = Logistic.loadDataSet()
# print(dataSet)
# print(labelSet)
weights = Logistic.gradAscent(dataSet,labelSet)
print(weights)
Logistic.plotBestFit(weights.getA())
# print(weights.getA())

# weights = Logistic.dtocGradAscent0(array(dataSet) , labelSet)
# print(weights)
weights = Logistic.stocGradscent1(array(dataSet),labelSet,500)
print(weights)
Logistic.plotBestFit(weights)
