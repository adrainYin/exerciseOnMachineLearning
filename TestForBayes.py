import Bayes


# postingList , labelVertex = Bayes.loadData2Vertex()
# vocaList = Bayes.createZVocaList(postingList)
# print(vocaList)
# print(vocaList.__len__())
#
# inputSet = ['you','how','Can','buying']
# returnVec = Bayes.setWords2Vec(vocaList,inputSet)
# print(returnVec)
#
# Bayes.testingBayes()
errorCount = 0.0
for i in range(10):
    error =  Bayes.spamText()
    errorCount += error
print('平均错误率是',float(errorCount / 10))
