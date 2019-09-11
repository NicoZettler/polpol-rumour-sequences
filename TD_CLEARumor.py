# -*- coding: utf-8 -*-
# Top-down RvNN implementation based on the model of Jing Ma et al.
# (https://github.com/majingCUHK/Rumor_RvNN ; state: 10.09.2019)
# to improve the results of the rumor verification accomplished by the CLEARumor approach by Ipek Baris et al.
# (https://github.com/Institute-Web-Science-and-Technologies/CLEARumor ; state: 10.09.2019).
# @authors: Dhurim Sylejmani and Nico Zettler

#import sys
#from importlib import reload
#reload(sys)

#import RvNN_model
#import math

#import theano
#from theano import tensor as T
#import numpy as np
#from numpy.testing import assert_array_almost_equal

#import time
#import datetime
#import random
#from evaluate import *

from warnings import filterwarnings
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.exceptions import UndefinedMetricWarning
filterwarnings('ignore', category=UndefinedMetricWarning)
import numpy as np
from model.data import load_data, load_labels
from model.preprocessing import calc_treeDic, trim_tree, fit_to_model
from model.treebuilding import load_tree_data
from model.model import establish_model

vocabulary_size = 9189 # todo: make this value smaller later (when vocabulary is smaller because of NLP)
hidden_dim = 100
Nclass = 3

# load data => find the labels for training and test data and put them into a dict structured as needed in RvNN approach
# at first handle zip files as in CLEARumor implementation
train_data, dev_data, test_data, twitter_train_data, twitter_test_data, \
    reddit_train_data, reddit_dev_data, reddit_test_data, training_data_archive, \
    test_data_archive = load_data()

# load all labels (train/dev/test) into one dictionary as (sourceID:label)
# and all IDs for training and test data into two separate lists
labelDic, indexDic, train_IDs, test_IDs, highest_source_eid = load_labels(train_data, dev_data, test_data)

# create treeDic as it is needed for the RvNN
treeDic = {}
word_index = 0 # give every word a unique index starting with 0
words = {} # dict containing words of all posts combined => ('word':word_index)
# twitter data
for archive, topics in [(training_data_archive, twitter_train_data.items()),
                            (test_data_archive, twitter_test_data.items())]:
    for topic, threads in topics:
        for thread in threads.values():
            treeDic, word_index, highest_source_eid = calc_treeDic(treeDic, thread, True, word_index, highest_source_eid, archive, indexDic, words)
# reddit data
for archive, threads in [(training_data_archive, reddit_train_data),
                            (training_data_archive, reddit_dev_data),
                            (test_data_archive, reddit_test_data)]:
    for thread in threads.values():
        treeDic, word_index, highest_source_eid = calc_treeDic(treeDic, thread, False, word_index, highest_source_eid, archive, indexDic, words)

# iterate through treeDic again and change parent indices to the corresponding smaller values of indexDic
# and also delete tree branches that contain posts which don't appear in the dataset
treeDic = trim_tree(treeDic, indexDic)

# load training data
tree_train, word_train, index_train, y_train, parent_num_train = load_tree_data(indexDic, labelDic, treeDic, train_IDs)
word_train, index_train, tree_train = fit_to_model(word_train, index_train, tree_train)

# load test data
tree_test, word_test, index_test, y_test, parent_num_test = load_tree_data(indexDic, labelDic, treeDic, test_IDs)
word_test, index_test, tree_test = fit_to_model(word_test, index_test, tree_test)

# establish RvNN model
model = establish_model(vocabulary_size, hidden_dim, Nclass)

def evaluate(y_test: list, prediction: list) -> None:
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
        y_pred.append(maxIdx)
    print("Accuracy: ", accuracy_score(y_truth, y_pred), " F1-Macro: ", f1_score(y_truth, y_pred, average='macro'))

# gradient descent
Nepoch = 50
learning_rate = 0.005
losses_5, losses = [], []
count_samples = 0
for epoch in range(Nepoch):
    indexs = [i for i in range(len(y_train))]
    for i in indexs:
        loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], learning_rate)
        print("iteration ", i)
        losses.append(np.round(loss,2))
        count_samples += 1
    print("Epoch: ", epoch, " Loss: ", np.mean(losses))
    
    #if epoch % 5 == 0: #PROVISORISCH: nachher wieder einrücken
    losses_5.append((count_samples, np.mean(losses)))
    prediction = []
    for j in range(len(y_test)):
        prediction.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j]))
    evaluate(y_test, prediction)
    ## Adjust the learning rate if loss increases
    if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
        learning_rate = learning_rate * 0.5   
        print("Setting learning rate to ", learning_rate)
    losses = []