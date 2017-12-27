import numpy
import scipy

def is_true(x,y):
    if x<y:
        return True
    else:
        return False

def feibonaqiehanshu(n):
    if n ==0:
        return 0
    elif n ==1:
        return 1
    else:
        return feibonaqiehanshu(n-1) + feibonaqiehanshu(n-2)

def printString(a):
    print(a)


def feindwords(words,letter):
    index  = 0
    while index < len(words):
        if words[index] == letter:
            return index
        else:
            index = index + 1

    return -1

def upperWords(words):
    return words.upper()


def findWordsFromWord1InWords2(words1,words2):
    for letter in words1:
        if letter in words2:
            print(letter)

def isequals(words1,words2):
    if words1 == words2:
        print('true')
    else:
        print('false')

def isReverse(words1,words2):
    if len(words1) != len(words2):
         return False
    i = 0
    j = len(words2) -1

    while j >= 0 :
        if words1[i] != words2[j]:
         return False
    i = i+1
    j = j-1
    return True

def capitalize(t):
    res = []
    for i in t:
        res.append(i.capitalize())
    return res
def deleteFirstElement(letters):
    del letters[0]