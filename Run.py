from numpy import *
import operator



def printFunction():
    print("you are a sha bi ")
    print("i am a  good man")

def getMetrix():
    metrix = array([[1,2,3],[3,4,2],[4,5,7]])
    return metrix
def getIntMetrix():
    intMetrix = array([4,3,6,5])
    return intMetrix


def compute(a,b):
    a = getIntMetrix()
    #metrixSize = a.shape[0]
    b = getMetrix()
    return tile(a,(3,1))

def sort(array):
    return array.argsort()

def getElement():
    element = array(['a','b','c','d'])
    return element
# element = getElement()
def checkElement(a):
    print(a + 'helloworld')
def checkElement():
    return 'abc'

# def sumElement():
#     print(Run.checkElement() + 'de')



# def getiteritems(dir):
#     it = dir.iteritems();
#     return it


def classify0(intX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX ,(dataSet,1)) - dataSet
    # tile重复代码行，将array 重读为一个二维的情况，行重复dataSet的大小，列重复为1
    sqDiffMAt = diffMat ** 2
    sqDistance = sqDiffMAt.sum(axis=1)
    # axis = 1 表示第二维度，设定维度从0开始
    #axis = 0表示每一列的相加，axis = 1表示每一行的相加
    distance = sqDistance **0.5
    sortdDistanceIndices = distance.argsort()
    #argsort函数返回一个数组，是从小到大顺序排列的序列号，构成一个array
    classCount = {}
    for i in range(k):
        voteLable = labels(sortdDistanceIndices[i])
        classCount = classCount.get(voteLable,0)+1
        sortClassCount = sorted(classCount.iteritems(),
        key = operator.itemgetter(1),reverse = True)

        return sortClassCount[0][0]

def printString(a):
    print(a)

