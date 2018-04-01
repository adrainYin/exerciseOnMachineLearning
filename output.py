import KNN
import decisionTree
import plotTree
# KNN.handwritingTest()

dataSet,label = decisionTree.creatData()
dataSet1 = decisionTree.file2list("CH03_data/watermelon.txt")
# resultSet = decisionTree.splitClass(dataSet,0,0)
# resultSet = decisionTree.splitClass(dataSet1,3,'didntLike')
# print(dataSet)
# gainEnt,featList = decisionTree.computeEntOnFeat(dataSet,0,[0,1])
# print(gainEnt)
# print(featList)
# print(dataSet1)
gainEnt,feature = decisionTree.computeEntOnFeat(dataSet1,0,['是','否'])
print(gainEnt)
print(feature)
print(decisionTree.computeEnt(dataSet1))
print(len(dataSet[0]))
print(dataSet[0])
print(decisionTree.chooseBestFeature(dataSet1))
reslut = decisionTree.majorityClass([1,1,3,3,3,4,4,2,2,1,1,1,5,5])
print(reslut)
tree =  decisionTree.createTree(dataSet1,['色泽','根蒂','敲声','纹理','部脐','触感'])
print(tree)
# print(plotTree.getTreeLeafNode(tree))
# # print(plotTree.getTreeDepth(tree))
# print(plotTree.getDepth(tree))
# plotTree.createPlot(tree)
mytree = {'no surfacring':{0:'no',1:{'flippers':{0:'no',1:'yes'}},3:'maybe'}}
# plotTree.createPlot(mytree)
featLbaels = ['色泽','根蒂','敲声','纹理','部脐','触感']
testList = ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑']
classify =  plotTree.classify(tree,featLbaels,testList)
print(classify)

decisionTree.storeTree(tree,"CH03_data/classifyTree.txt")
print(decisionTree.getTree("CH03_data/classifyTree.txt"))

dir = [{'my':'vczh is the god','you':'I am a god too'},'helloshanghai',1,4.554]
decisionTree.storeTree(dir,"CH03_data/testJSON.txt")
print(decisionTree.getTree("CH03_data/testJSON.txt"))