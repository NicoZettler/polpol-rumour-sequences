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

import TD_RvNN_exp
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

# create treeDic as it is needed for the RvNN
treeDic = {}
for archive, topics in [(training_data_archive, twitter_english.items()),
                            (test_data_archive, twitter_en_test_data.items())]:
    for topic, threads in topics:
            for thread in threads.values():
                # get the source information as a Dict (contains all info about the source post)
                source_information = json.loads(archive.read(list(thread['source-tweet'].values())[0]))
                idx = source_information['id'] # get the 18 digit source ID
                eid = indexDic[str(idx)] # convert it to the corresponding simpler ID
                post_structure = json.loads(archive.read(thread['structure.json'])) # get the thread structure as a Dict
                parent_num = calc_parent_num(post_structure) # calculate the number of reply levels in each thread structure
                #print(eid, parent_num,  " PAUSE ")
                indexC = eid # initialize post index with source post index
                if idx not in treeDic: # create empty entry first to make the key accessable
                    treeDic[idx] = {}
                treeDic[idx][indexC] = {'parent':'None', 'parent_num':parent_num}
                if 'replies' in thread: # some "replies" folders seem to be empty and then this for loop would throw an error
                    for reply in thread['replies'].values(): # for every reply post
                        # get the reply information as a Dict (contains all info about the reply post)
                        reply_information = json.loads(archive.read(reply))
                        reply_idx = reply_information['id']
                        indexC = highest_source_eid # convert 18 digit index to simpler index
                        indexDic[str(reply_idx)] = highest_source_eid # save connection between IDs (18 digit and simple)
                        # RvNN implementation; maybe change variable names later for better understanding
                        # find out parent of each reply node
                        indexP = find_parent_node(post_structure, str(reply_idx)) # somehow the "key" input has to be a string
                        #print(indexC, indexP, parent_num)
                        treeDic[idx][indexC] = {'parent':indexP, 'parent_num':parent_num} # put everything at the right place
                        highest_source_eid += 1

# iterate through treeDic again and change parent indices to the corresponding smaller values of indexDic
# and also delete tree branches that contain posts which don't appear in the dataset
for i,v in treeDic.items():
    for j,w in list(v.items()):
        if not w['parent'] == 'None':
            if w['parent'] in indexDic:
                #print(w['parent'], " before")
                w['parent'] = indexDic[w['parent']]
                #print(w['parent'], " after")
            else: # delete all branches in the tree which start with IDs that don't appear in the actual data 
                print(w['parent'], w) # (because there is no statistical significance if we can't track the posts
                print("whole branch ", treeDic[i][j])   # they refer to)
                del treeDic[i][j]
#print(treeDic)

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
        node = TD_RvNN_exp.Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j 
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        #wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
        #print tree[j]['maxL']
        #nodeC.index = wordIndex
        #nodeC.word = wordFreq
        #nodeC.time = tree[j]['post_t']
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
    #ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
    #x_word, x_index, tree = tree_gru_u2b.gen_nn_inputs(root, ini_x, ini_index) 
    tree = TD_RvNN_exp.gen_nn_inputs(root)
    return tree, parent_num

# load training data
tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
l1,l2,l3 = 0,0,0
for eid in train_IDs:
    print(eid)
    if indexDic[eid] not in labelDic: continue
    if int(eid) not in treeDic: continue # somehow must be casted to int
    if len(treeDic[int(eid)]) <= 0:
        continue
    label = labelDic[indexDic[eid]] # somehow eid must be referenced as string to adress the key
    y, l1,l2,l3 = loadLabel(label, l1, l2, l3)
    y_train.append(y)
    tree, parent_num = constructTree(treeDic[int(eid)])
    tree_train.append(tree)
    parent_num_train.append(parent_num)
    c += 1
    print(tree)
print(l1, l2, l3)

# load test data
tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
l1,l2,l3 = 0,0,0
for eid in test_IDs:
    print(eid)
    if indexDic[eid] not in labelDic: 
        print("1")
        continue
    if int(eid) not in treeDic:
        print("2")
        continue
    if len(treeDic[int(eid)]) <= 0:
        print("3")
        continue
    label = labelDic[indexDic[eid]]
    y, l1,l2,l3 = loadLabel(label, l1, l2, l3)
    y_test.append(y)
    ## 2. construct tree
    tree, parent_num = constructTree(treeDic[int(eid)])
    tree_test.append(tree)
    parent_num_test.append(parent_num)
    c += 1
    print(tree)
print(l1,l2,l3)
print("train no:", len(tree_train), len(parent_num_train), len(y_train))
print("test no:", len(tree_test), len(parent_num_test), len(y_test))
print("dim1 for 0:", len(tree_train[0]))
print("case 0:", tree_train[0][0], parent_num_train[0])
