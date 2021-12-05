import nltk
from nltk.tokenize import sent_tokenize  
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re

# load the file
train_data = open('sample_train.txt').read().lower()

#print(len(train_data))
#split per sentence
sentences = sent_tokenize(train_data)
# splits every word in each sentence
tokenizer = RegexpTokenizer("[\w']+")
# adds a tag to each wrd
lsts = [nltk.pos_tag(tokenizer.tokenize(sentence)) for sentence in sentences]
# arr for transition matrix
arr = []
for lst in lsts:
    tmp = []
    for item in lst:
        tmp.append(item[1])
    arr.append(tmp)

#print(len(train_data))

test_data = open('sample_test.txt').read().lower()

#print(len(test_data))


tSentences = sent_tokenize(test_data)

tokenizer = RegexpTokenizer("[\w']+")

# tagger of test_list
test_lst = [nltk.pos_tag(tokenizer.tokenize(sentence)) for sentence in tSentences]



tagger = []
for lst in test_lst:
    tmp = []
    for item in lst:
        tmp.append(item[1])
    tagger.append(tmp)


#list of list containing lists of words cleaned
lsts2 = [tokenizer.tokenize(sentence) for sentence in tSentences]

# results in a list of lists containing sentences
lst3 = []
for lst in lsts2:
   lst3.append(" ".join(lst))













