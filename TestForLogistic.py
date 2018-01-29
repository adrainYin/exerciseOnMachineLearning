import Logistic
dataSet , labelSet = Logistic.loadDataSet()
print(dataSet)
print(labelSet)
weights = Logistic.gradAscent(dataSet,labelSet)
print(weights)
Logistic.plotBestFit(weights.getA())
print(weights.getA())