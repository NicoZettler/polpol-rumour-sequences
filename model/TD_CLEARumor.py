# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: RvNN implementation for the CLEARumor dataset
@author: nicozettler
@structure: RvNN
@variable: Nepoch, lr, obj, fold
@time: July 28, 2019
"""

import sys
from importlib import reload
reload(sys)

import TD_RvNN
import math

import theano
from theano import tensor as T
import numpy as np
from numpy.testing import assert_array_almost_equal

import time
import datetime
import random
from evaluate import *

# paths for CLEARumor training and test data
trainPath = "../resource/rumoureval-2019-training-data.zip"
testPath = "../resource/rumoureval-2019-test-data.zip"
testLabelPath = "../resource/final-eval-key.json" # evaluation file contains test labels

# load data => find the labels for training and test data and put them into a dict structured as needed in RvNN approach
# at first handle zip files as in CLEARumor implementation
from zipfile import ZipFile
from typing import Dict
import json
def get_archive_directory_structure(archive: ZipFile) -> Dict:
    result = {}
    for file in archive.namelist():
        # Skip directories in archive.
        if file.endswith('/'):
            continue

        d = result
        path = file.split('/')[1:]  # [1:] to skip top-level directory.
        for p in path[:-1]:  # [:-1] to skip filename
            if p not in d:
                d[p] = {}
            d = d[p]
        d[path[-1]] = file
    return result
    
training_data_archive = ZipFile(trainPath)
training_data_contents = get_archive_directory_structure(
        training_data_archive)
train_labels = json.loads(training_data_archive.read(training_data_contents['train-key.json']))
dev_labels = json.loads(training_data_archive.read(training_data_contents['dev-key.json']))
test_labels = json.load(open(testLabelPath,'r'))

# load all labels (train/dev/test) into one dictionary as (sourceID:label)
# and all IDs for training and test data into two separate lists
labelDic, indexDic = {}, {} # labelDic contains all (eid,label) connections and indexDic contains (idx, eid) translation
# to make indexing more simple and avoid 18 digit numbers
eid = 1 # start indexing at one and assign each new tweet an index eid+=1        
train_IDs, test_IDs = [], []
for (idx, label) in train_labels['subtaskbenglish'].items():
    if len(idx) == 18: # for now only consider the twitter IDs and labels
        indexDic[idx] = eid # keep connection between simple index and 18 digit index for look-ups later
        labelDic[eid] = label.lower()
        train_IDs.append(idx)
        eid += 1 # increase index by one for the next tweet
for (idx, label) in dev_labels['subtaskbenglish'].items():
    if len(idx) == 18: # for now only consider the twitter IDs and labels
        indexDic[idx] = eid
        labelDic[eid] = label.lower()
        train_IDs.append(idx)
        eid += 1
for (idx, label) in test_labels['subtaskbenglish'].items():
    if len(idx) == 18: # for now only consider the twitter IDs and labels
        indexDic[idx] = eid
        labelDic[eid] = label.lower()
        test_IDs.append(idx)
        eid += 1
highest_source_eid = eid # keep this value to continue counting upwards for simpler reply indices later

# generate the tree from the zip file data
twitter_english = training_data_contents['twitter-english']
test_data_archive = ZipFile(testPath)
test_data_contents = get_archive_directory_structure(test_data_archive)
twitter_en_test_data = test_data_contents['twitter-en-test-data']

# calculate parent_num, indexP and indexC for the treeDic
def calc_parent_num(tree_branch: Dict) -> int: # go recursively through tree and calculate parent_num
    if isinstance(tree_branch, Dict):
        return 1 + (max(map(calc_parent_num, tree_branch.values())) if tree_branch else 0)
    return 0
    
def find_parent_node(tree_branch: Dict, reply_idx: int) -> int: # go recursively through the tree and find parent index
    for indexP, sub_branch in tree_branch.items(): # keep indexP and go through the keys
        if reply_idx in sub_branch: # search for reply_idx in the keys
            return indexP # if it is found, return the parent node
        elif isinstance(sub_branch, dict): # else check if the keys itself are a dictionary
            parent_node = find_parent_node(sub_branch, reply_idx) # if so, recursively call again the function with the subtree 
            if parent_node is not None: # return the value only if it exists (otherwise we should add try and catch later)
                return parent_node

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
#print(len(stop_words))
stop_words.extend([":", ",", ".", "#", "@", "-", "(", ")", ";", "&", "'"])
#print(len(stop_words))
#print(stop_words)
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# removes unimportant constructs, tokenizes and stems a (str) post and turns it into a list of (str) words
def preprocess_words(raw_post: str) -> list:
    post_words = []
    for word in raw_post.split():
        if not word.startswith('@') and not word.startswith('http'): # remove mentions and URLs
            post_words.append(word)
        elif word.startswith('http'):
            post_words.append('http') # for each link put http such that URLs can be identified for the indexing
    clean_post = ' '.join(post_words) # put post back together
    text_information = word_tokenize(clean_post) # 
    text_information = [ps.stem(word) for word in text_information] # stem all words
    return text_information
    
def calc_word_index_freq_pairs(text_information: list, word_index: int, maxL: int) -> (str, int, int):
    Vec = str()
    words_per_post = {} # dict with frequencies for every word in a post; reset for every post
    count_post_length = 0 # reset for every loop iteration
    for word in text_information: # iterate through words of post
        if word not in stop_words: # make sure not to include stop words, because they don't have any relevance

            if word not in words.keys():
                words[word] = word_index # give every word a unique index
                word_index += 1
            count_post_length += 1

            if word not in words_per_post.keys():
                words_per_post[word] = 1
            else:
                words_per_post[word] += 1

    iteration = 0 # count iterations to get the last iteration and not put ' ' at the end of Vec
    for word in words_per_post.keys(): # iterate through words of post a second time to get the right numbers

        Vec += str(words[word]) + ':' + str(words_per_post[word])
        iteration += 1
        if iteration != len(words_per_post.keys()): # if it's not the end of the tweet
            Vec += ' ' # add space between the word index/frequency pairs
            
    if maxL < count_post_length: # new maximum post length found
        maxL = count_post_length
        
    return Vec, word_index, maxL

# create treeDic as it is needed for the RvNN
treeDic = {}
word_index = 0 # give every word a unique index starting with 0
words = {} # dict containing words of all posts combined => ('word':word_index)
for archive, topics in [(training_data_archive, twitter_english.items()),
                            (test_data_archive, twitter_en_test_data.items())]:
    for topic, threads in topics:
        for thread in threads.values():
            maxL = 0 # maximum post length of each thread => reset to 0 for each new thread
            # get the source information as a Dict (contains all info about the source post)
            source_information = json.loads(archive.read(list(thread['source-tweet'].values())[0]))
            idx = source_information['id'] # get the 18 digit source ID
            eid = indexDic[str(idx)] # convert it to the corresponding simpler ID
            post_structure = json.loads(archive.read(thread['structure.json'])) # get the thread structure as a Dict
            parent_num = calc_parent_num(post_structure) # calculate the number of reply levels in each thread structure
            indexC = eid # initialize post index with source post index
            
            # preprocessing of current source post
            text_information = preprocess_words(source_information['text'])
            #print(text_information)
            
            Vec, word_index, maxL = calc_word_index_freq_pairs(text_information, word_index, maxL)
            #print(Vec, " wi: ", word_index, " maxL: ", maxL)
            
            if idx not in treeDic: # create empty entry first to make the key accessable
                treeDic[idx] = {}
            treeDic[idx][indexC] = {'parent':'None', 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
            if 'replies' in thread: # some "replies" folders seem to be empty and then this for loop would throw an error
                for reply in thread['replies'].values(): # for every reply post
                    # get the reply information as a Dict (contains all info about the reply post)
                    reply_information = json.loads(archive.read(reply))
                    reply_idx = reply_information['id']
                    indexC = highest_source_eid # convert 18 digit index to simpler index
                    indexDic[str(reply_idx)] = highest_source_eid # save connection between IDs (18 digit and simple)
                    # find out parent of each reply node
                    indexP = find_parent_node(post_structure, str(reply_idx))
                    
                    # preprocessing of current reply post
                    text_information = preprocess_words(reply_information['text'])
                    #print(text_information)
                    
                    Vec, word_index, maxL = calc_word_index_freq_pairs(text_information, word_index, maxL)
                    #print(Vec, " wi: ", word_index, " maxL: ", maxL)
                    
                    treeDic[idx][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
                    highest_source_eid += 1 # increase index for the next reply

            for post in treeDic[idx].values(): # go through all posts again to set the maxL for every thread
                post['maxL'] = maxL

# iterate through treeDic again and change parent indices to the corresponding smaller values of indexDic
# and also delete tree branches that contain posts which don't appear in the dataset
for i,v in treeDic.items():
    for j,w in list(v.items()):
        if not w['parent'] == 'None':
            if w['parent'] in indexDic:
                w['parent'] = indexDic[w['parent']]
            else: # delete all branches in the tree which start with IDs that don't appear in the actual data 
                #print(w['parent'], w) # (because there is no statistical significance if we can't track the posts
                #print("whole branch ", treeDic[i][j])   # they refer to)
                del treeDic[i][j]
            if w['vec'] == '':
                #print(w['parent'], w)
                #print("vec whole branch ", treeDic[i][j])
                del treeDic[i][j]
#print(treeDic)

# function for handling word frequencies and indices
def str2matrix(Str, MaxL): # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [ 0 for i in range( MaxL-l ) ]
    wordFreq += ladd 
    wordIndex += ladd 
    return wordFreq, wordIndex

# load label function
def loadLabel(label, l1, l2, l3):
    labelset_f, labelset_t, labelset_u = ['false'], ['true'], ['unverified']
    if label in labelset_f:
       y_train = [1,0,0]
       l1 += 1
    if label in labelset_t:
       y_train = [0,1,0]
       l2 += 1
    if label in labelset_u:
       y_train = [0,0,1]
       l3 += 1
    return y_train, l1,l2,l3


# construct tree function
def constructTree(tree):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node
    index2node = {}
    for i in tree:
        node = TD_RvNN.Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ## 
        if not indexP == 'None':
            if int(indexP) in index2node: # there are two IDs listed in the structure files which don't exist in the training set
                nodeP = index2node[int(indexP)]       # => filter them out
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
    ## 3. convert tree to DNN input    
    parent_num = tree[j]['parent_num'] 
    ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
    x_word, x_index, tree = TD_RvNN.gen_nn_inputs(root, ini_x) # put root node (with all children) and 0 array with maxL
    return x_word, x_index, tree, parent_num

# load training data
tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
l1,l2,l3 = 0,0,0
for eid in train_IDs:
    if indexDic[eid] not in labelDic: continue
    if int(eid) not in treeDic: continue
    if len(treeDic[int(eid)]) <= 0:
        continue
    label = labelDic[indexDic[eid]]
    y, l1,l2,l3 = loadLabel(label, l1, l2, l3)
    y_train.append(y)
    x_word, x_index, tree, parent_num = constructTree(treeDic[int(eid)])
    tree_train.append(tree)
    word_train.append(x_word)
    index_train.append(x_index)
    parent_num_train.append(parent_num)
    c += 1

# load test data
tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
l1,l2,l3 = 0,0,0
for eid in test_IDs:
    if indexDic[eid] not in labelDic:
        continue
    if int(eid) not in treeDic:
        continue
    if len(treeDic[int(eid)]) <= 0:
        continue    
    label = labelDic[indexDic[eid]]
    y, l1,l2,l3 = loadLabel(label, l1, l2, l3)
    y_test.append(y)
    ## 2. construct tree
    x_word, x_index, tree, parent_num = constructTree(treeDic[int(eid)])
    tree_test.append(tree)
    word_test.append(x_word)
    index_test.append(x_index)
    parent_num_test.append(parent_num)
    c += 1
print(l1,l2,l3)
print("train no:", len(tree_train), len(parent_num_train), len(y_train))
print("test no:", len(tree_test), len(parent_num_test), len(y_test))
print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
print("case 0:", tree_train[0], word_train[0][0], index_train[0][0],  parent_num_train[0])

# RvNN testing
vocabulary_size = 7588
hidden_dim = 100
Nclass = 3
t0 = time.time()
model = TD_RvNN.RvNN(vocabulary_size, hidden_dim, Nclass)
t1 = time.time()
print('Recursive model established,', (t1-t0)/60)

# adapt word_train data (pad zeros such that numpy does not throw any errors)
for j in range(0,len(word_train)):
    word_ext = []
    for i in range(0,10000):
        if i < len(word_train[j]):
            word_ext.append(word_train[j][i])
        else:
            word_ext.append([0] * len(word_train[j][0])) # pad lines of zeros
    word_train[j] = word_ext
# and also index_train and tree_train (because some trees are too small to handle)
for i in range(0,len(index_train)):
    index_ext = list(index_train[i])
    tree_ext = list(tree_train[i])
    while len(index_ext) < 10: # pad zeros because RvNN doesn't work with smaller arrays
        index_ext.append([0] * len(index_train[i][0]))
        tree_ext.append([0] * 2)
    index_train[i] = index_ext
    tree_train[i] = tree_ext
# same with test data
for j in range(0,len(word_test)):
    word_ext = []
    for i in range(0,10000):
        if i < len(word_test[j]):
            word_ext.append(word_test[j][i])
        else:
            word_ext.append([0] * len(word_test[j][0])) # pad lines of zeros
    word_test[j] = word_ext
for i in range(0,len(index_test)):
    index_ext = list(index_test[i])
    tree_ext = list(tree_test[i])
    while len(index_ext) < 10: # pad zeros because RvNN doesn't work with smaller arrays
        index_ext.append([0] * len(index_test[i][0]))
        tree_ext.append([0] * 2)
    index_test[i] = index_ext
    tree_test[i] = tree_ext

# Gradient descent

from sklearn.metrics import accuracy_score, classification_report, f1_score

Nepoch = 50 # change to a higher number later
lr = 0.005
losses_5, losses = [], []
num_examples_seen = 0
for epoch in range(Nepoch):
    indexs = [i for i in range(len(y_train))]
    for i in indexs:
        loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], lr)
        #print("iteration ", i)
        losses.append(np.round(loss,2))
        num_examples_seen += 1
    print("epoch=%d: loss=%f" % ( epoch, np.mean(losses) ))
    sys.stdout.flush()
    
    ## cal loss & evaluate
    if epoch % 5 == 0: #PROVISORISCH: nachher wieder einrücken
        losses_5.append((num_examples_seen, np.mean(losses))) 
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))    
        sys.stdout.flush()
        prediction = []
        for j in range(len(y_test)):
            prediction.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j]))
        #res = evaluation_3class(prediction, y_test)
        #PROVISORISCH
        y_truth = []
        y_pred = []
        for i in range(0,len(y_test)):
            if y_test[i][0] == 1:
                y_truth.append(0)
            elif y_test[i][1] == 1:
                y_truth.append(1)
            else:
                y_truth.append(2)
            maxim = 0
            maxIdx = 3
            for j in range(0,3):
                if list(prediction[i][0])[j] > maxim:
                    maxim = list(prediction[i][0])[j]
                    maxIdx = j
                #print(list(prediction[i][0])[j])
            #print(maxim, maxIdx)
            y_pred.append(maxIdx)
        print("Accuracy: ", accuracy_score(y_truth, y_pred), " F1-Macro: ", f1_score(y_truth, y_pred, average='macro'))
        #PROVISORISCH
        #print('results:', res)
        sys.stdout.flush()
        ## Adjust the learning rate if loss increases
        if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
            lr = lr * 0.5   
            print("Setting learning rate to %f" % lr)
            sys.stdout.flush()
    losses = []
