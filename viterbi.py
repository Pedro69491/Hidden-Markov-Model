import numpy as np
from collections import OrderedDict
import parser
import time
tags = ['LS', 'TO', 'VBN', 'WP', 'UH', 'VBG', 'JJ', 'VBZ', 'VBP', 'NN', 'DT', 'PRP', 'WP$', 'NNPS', 'PRP$', 'WDT', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']


# list containing all the tags from training data words
tagsList  = parser.arr



# how to fetch a certain percentage of the training data
#tagsList = tagsList[0:round(len(tagsList) * 0.9)]
#print(len(tagsList))
wrdTgList = parser.lsts
#wrdTgList = wrdTgList[0:round(len(wrdTgList) * 0.9)]
#list of lists containing sentences
test_list = parser.lst3
#print(test_list)
# test list tags
testTagsList = parser.tagger
#print(len(testTagsList))
matrix1 = np.zeros((36, 36))

# number of times each tag appears in the in training data
def countTag(tags, sentences):
    count = 0
    for i in tags:
        lst = [wrd for sentence in sentences for wrd in sentence if i == wrd]
        count = len(lst)            
        transitionMatrix(i, count)


def transitionMatrix(tag, denominator):
    count = 0
    if denominator == 0:
        for i in range(len(tags)):
            matrix1[tags.index(tag)][i] = 0.001
    else:
        for i in range(len(tags)):
            count = 0
            for sentence in tagsList:
                for j in range(len(sentence)-1):
                    if tag == sentence[j] and tags[i] == sentence[j+1]:
                        count += 1
            if count == 0:
                matrix1[tags.index(tag)][i] = 0.001 
            else:
                matrix1[tags.index(tag)][i] = count/denominator

countTag(tags, tagsList)


counter = 0
lstWords = [] 
for sentence in wrdTgList:
    for dic in sentence:
        if dic[0] not in lstWords:
            lstWords.append(dic[0])
            counter += 1
            
matrix2 = np.zeros((36, counter))

# number of times each tag appears in the text
def countTotal(tag):
    totalCount = 0
    for sentence in tagsList:
        for tg in sentence:
            if tg == tag:
                totalCount += 1
    return totalCount


def emissionMatrix():
    dic = OrderedDict()
    count = 0
    for sentence in wrdTgList:
        for j in range(len(sentence)):
            key = sentence[j][0] + " " + sentence[j][1]
            if key not in dic:
                dic[key] = 1
            else:
                dic[key] += 1
    prevItems = []
    for k in dic:
        arr = k.split(' ')
        if arr[1] not in tags:
            continue
        
        if count == 0:
            matrix2[tags.index(arr[1])][count] = dic[k]/countTotal(arr[1])
        elif arr[0] not in prevItems:
            matrix2[tags.index(arr[1])][lstWords.index(arr[0])] = dic[k]/countTotal(arr[1])
        else:
            matrix2[tags.index(arr[1])][lstWords.index(arr[0])] = dic[k]/countTotal(arr[1])           
        prevItems.append(arr[0])
        count = 1

emissionMatrix()

# number of times a tag appears as initial value
def asInitialValue(tag):
    count = 0
    for sentence in tagsList:
        if len(sentence) < 1:
            continue
        if sentence[0] == tag:
            count += 1
    if countTotal(tag) == 0:
        return 0.001
    elif count == 0:
        return 0.001
    return count/countTotal(tag)


def checkWord(wrd):
    status = False
    for word in lstWords:
        if wrd == word:
            status = True
            return status
    return status


# viterbi method
def viterbi(num_tags, sentence):
    pathMatrix = np.zeros((num_tags, len(sentence)))
    for i in range(num_tags):
        if checkWord(sentence[0]):
            pathMatrix[i][0] = asInitialValue(tags[i]) * matrix2[i][lstWords.index(sentence[0])]
        else:
            pathMatrix[i][0] = asInitialValue(tags[i])
    for i in range(1, len(sentence)):
        for j in range(num_tags):
            maxTmp = -np.inf
            for z in range(num_tags):
                tmp = pathMatrix[z][i-1] * matrix1[z][j]
                if maxTmp < tmp:
                    maxTmp = tmp
                    if checkWord(sentence[i]):
                        pathMatrix[j][i] = maxTmp * matrix2[j][lstWords.index(sentence[i])]
                    else:
                        pathMatrix[j][i] = maxTmp 
    answer = []
    for arr in pathMatrix.T:
        index = np.where(arr == np.max(arr))
        answer.append(tags[index[0][0]])
    return answer






def tokenAccuracyFirstWord(list1, list2):
    count = 0
    if list1[len(list1)-1] == list2[len(list1)-1]:
        count = 1
    return count

def tokenAccuracy(list1, list2):
    count = 0
    num_elem = len(list1)
    for i in zip(list1, list2):
            if i[0] == i[1]:
                count += 1
    return count/num_elem


bool_lst = []
def sentenceAccuracy(list1, list2):
    status_sent = True
    for i in zip(list1, list2):
        if i[0] != i[1]:
            status_sent = False
            break
    if status_sent == True:
        bool_lst.append(status_sent)
        
accum2 = 0
accum = 0
for i in range(len(test_list)):
    sentence = test_list[i].split()
    result = viterbi(36, sentence)
    print(result)
    print(testTagsList[i])
    accum2 += tokenAccuracyFirstWord(result, testTagsList[i])
    accum += tokenAccuracy(result, testTagsList[i])
    sentenceAccuracy(result, testTagsList[i])
print(accum2/len(test_list))
print("Token accuracy: " + str(accum/len(test_list)))
print("Full sentence accuracy: " + str(len(bool_lst)/len(test_list)))


 
