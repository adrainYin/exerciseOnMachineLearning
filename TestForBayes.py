import Bayes
import feedparser


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
# errorCount = 0.0
# for i in range(10):
#     error =  Bayes.spamText()
#     errorCount += error
# print('平均错误率是',float(errorCount / 10))
#
# string = open('CH04_data/spam/1.txt').read()
# list = ['have','for']
# print(Bayes.calcMostFreq(list,string))

#对网上数据集的采样
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# count = 0.0
# for i in range(10):
#     vocabList,pSF,pNY,error = Bayes.localWords(ny,sf)
#     count += error
#     print('第%d次的pSF和pNY分别是' % i , pSF ,  '  ', pNY)
# print(float(count / 10))
# print(vocabList)
Bayes.getTopWords(ny,sf)