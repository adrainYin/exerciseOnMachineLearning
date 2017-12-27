import matplotlib.pyplot as plt
# import pyplotz.pyplotz as plt
import numpy

decisionNode = dict(boxstyle="sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle="<-")

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon = False)
    plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()



def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )



def getTreeLeafNode(tree):
    numLeafNode = 0
    keylist = list(tree.keys())  #第一层的字典的key的长度总是1
    # print("list的长度")
    # print(len(keylist))
    firstStr = keylist[0] #第一层是所有子节点的结合
    secondDir = tree[firstStr]  #第二层是所有
    for key in secondDir.keys():
        if type(secondDir[key]) == dict:
            numLeafNode += getTreeLeafNode(secondDir[key])
        else:
            numLeafNode +=1
    return numLeafNode

def getTreeDepth(tree):
    maxdepth = 0
    keyList = list(tree.keys())
    firstStr = keyList[0]
    secondDir = tree[firstStr]
    for key in secondDir.keys():
        if type(secondDir[key]) == dict:  #是第二层的属性取值对应的分类
            thisdepth = 1 + getTreeDepth(secondDir[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth:
            maxdepth = thisdepth
    return maxdepth

def getDepth(tree):
    maxdepth = 0
    keyList = list(tree.keys())
    firstDictKey = keyList[0]
    secondDict = tree[firstDictKey]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            thisdepth = getDepth(secondDict[key]) + 1
        else:
            thisdepth = 1
        if thisdepth > maxdepth:
            maxdepth = thisdepth
    return maxdepth

# def plotMidText(cntrPt,parentPt,txtString):
#     xMid = (parentPt[0] - cntrPt[0]) / 2.0 +cntrPt[0]
#     yMid = (parentPt[1] - cntrPt[1]) / 2,0 + cntrPt[1]
#     createPlot.ax1.text(xMid,yMid,txtString)

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(tree,parentPt,nodeTxt):
    numLeafs = getTreeLeafNode(tree)
    depth = getTreeDepth(tree)
    firstStrList = list(tree.keys())
    firstStr = firstStrList[0] #获取当前树的树根节点
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = tree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff , plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111,frameon = False, ** axprops)
    plotTree.totalW = float(getTreeLeafNode(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff =  float(-0.5 / plotTree.totalW)
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


# testData是特征标签，testLabels是特征标签的具体取值
def classify(intputTree,testData,testLabels):
    firstList = list(intputTree.keys())
    firstDict = firstList[0]
    secondDict = intputTree[firstDict]
    featIndex = testData.index(firstDict)
    for key in secondDict.keys():
        if key == testLabels[featIndex]:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key],testData,testLabels)
            else:
                classLabel = secondDict[key]
    return classLabel