import simpleBayse

dataSet = simpleBayse.file2list("CH03_data/watermelon.txt")
print(dataSet)
# print(classifyList)
# print(len(classifyList))
testVel = ['青绿',    '蜷缩',    '浊响' ,   '清晰' ,   '凹陷',    '硬滑','是']
dictVal,classifyList = simpleBayse.classify(dataSet,testVel)
print(dictVal)
print(classifyList)