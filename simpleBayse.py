import numpy
import operator

def file2list(filename):
    fr = open(filename)
    dataLines = fr.readlines()
    dataSet = []
    for lines in dataLines:
        lines = lines.strip()
        dataList = lines.split()
        dataSet.append(dataList)
    return dataSet



"""
朴素贝叶斯分类法，加入了拉普拉斯的修正准则
"""
def classify(dataSet,testVec):
    numDataSet = len(dataSet)

    featureNum = len(dataSet[0])-1
    labelPred = {}
    classifyList = []
    for i in range(len(dataSet[0])):
        oneFeatureDict = {}
        for j in range(numDataSet):
            if dataSet[j][i] not in oneFeatureDict.keys():
                oneFeatureDict[dataSet[j][i]] = 0
            oneFeatureDict[dataSet[j][i]] += 1
        classifyList.append(oneFeatureDict)

    classLabelDict = classifyList[-1]
    prodForLabel = {}
    for labelDict in classLabelDict.keys():
        #初始化标签分类函数
        predLabel = float((classLabelDict[labelDict] + 1) / (numDataSet + len(classLabelDict)))
        for i in range(featureNum):
            count = 0
            valNumForFeature = len(classifyList[i])
            for dataList in dataSet:
                if (dataList[i] == testVec[i]) and (dataList[-1] == labelDict):
                    count +=1
            featurePred = float((count + 1) / (classLabelDict[labelDict] + valNumForFeature))
            predLabel *= featurePred
        prodForLabel[labelDict] = predLabel

    prodForLabel = sorted(prodForLabel.items(), key=operator.itemgetter(1),reverse=True)

    return prodForLabel,classifyList


