import Bayes


postingList , labelVertex = Bayes.loadData2Vertex()
vocaList = Bayes.createZVocaList(postingList)
print(vocaList)
print(vocaList.__len__())

inputSet = ['you','how','Can','buying']
returnVec = Bayes.setWords2Vec(vocaList,inputSet)
print(returnVec)

Bayes.testingBayes()