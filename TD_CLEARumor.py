# Top-down RvNN implementation based on the model of Jing Ma et al.
# (https://github.com/majingCUHK/Rumor_RvNN ; state: 10.09.2019)
# to improve the results of the rumor verification accomplished by the CLEARumor approach by Ipek Baris et al.
# (https://github.com/Institute-Web-Science-and-Technologies/CLEARumor ; state: 10.09.2019).
# @authors: Dhurim Sylejmani and Nico Zettler

from warnings import filterwarnings
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.exceptions import UndefinedMetricWarning
from math import sqrt
import numpy as np
from model.data import load_data, load_labels
from model.preprocessing import estimate_word_frequencies, calc_tree_dic
from model.treebuilding import load_tree_data
from model.model import establish_model

filterwarnings('ignore', category=UndefinedMetricWarning)
filterwarnings('ignore', category=UserWarning)

# RvNN parameters
vocabulary_size = 5000
hidden_dim = 100
repitition_count = 10
class_count = 3
epoch_count = 51
learning_rate = 0.01

# Load data function:
# Split it into train, dev and test data as well as Twitter and Reddit data.
# Load also the .zip file structure in training and test data archive.
train_data, dev_data, test_data, twitter_train_data, twitter_test_data, \
    reddit_train_data, reddit_dev_data, reddit_test_data, training_data_archive, \
    test_data_archive = load_data()

# Load all labels (train/dev/test) into one dictionary as (sourceID:label)
# and all IDs for training and test data into three separate lists.
label_dic, train_IDs, dev_IDs, test_IDs = load_labels(train_data, dev_data, test_data)

# Calculate word frequencies.
words = {} # Dict containing words of all posts combined => ('word':word_index)
# Twitter data:
for archive, topics in [(training_data_archive, twitter_train_data.items()),
                            (test_data_archive, twitter_test_data.items())]:
    for topic, threads in topics:
        for thread in threads.values():
            words = estimate_word_frequencies(archive, thread, words, True)
# Reddit data:
for archive, threads in [(training_data_archive, reddit_train_data),
                            (training_data_archive, reddit_dev_data),
                            (test_data_archive, reddit_test_data)]:
    for thread in threads.values():
        words = estimate_word_frequencies(archive, thread, words, False)

# Sort words by frequency from highest to lowest.
sorted_words = sorted(words, key=words.get, reverse=True)[0:5000]

# Save all information needed for the RvNN in tree_dic dictionary.
tree_dic = {}
word_index = 0 # Give every word a unique index starting with 0.
# Twitter data:
for archive, topics in [(training_data_archive, twitter_train_data.items()),
                            (test_data_archive, twitter_test_data.items())]:
    for topic, threads in topics:
        for thread in threads.values():
            tree_dic, word_index = calc_tree_dic(tree_dic, thread, True, word_index, archive, sorted_words)
# Reddit data:
for archive, threads in [(training_data_archive, reddit_train_data),
                            (training_data_archive, reddit_dev_data),
                            (test_data_archive, reddit_test_data)]:
    for thread in threads.values():
        tree_dic, word_index = calc_tree_dic(tree_dic, thread, False, word_index, archive, sorted_words)

# Load training data:
tree_train, word_train, index_train, y_train, parent_num_train = load_tree_data(label_dic, tree_dic, train_IDs)

# Load dev data:
tree_dev, word_dev, index_dev, y_dev, parent_num_dev = load_tree_data(label_dic, tree_dic, dev_IDs)

# Load test data:
tree_test, word_test, index_test, y_test, parent_num_test = load_tree_data(label_dic, tree_dic, test_IDs)

# Evaluate function calculates accuracy, F1-score and RMSE depending on the predictions.
def evaluate(y_test: list, prediction: list) -> (list, list, list):
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
    acc = accuracy_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred, average='macro')
    rmse = sqrt(mean_squared_error(y_truth, y_pred))
    print("Accuracy: ", acc, " F1-Macro: ", f1, "RMSE: ", rmse)
    return acc, f1, rmse

# Save all scores in lists for evaluation.
accs_val, f1s_val, rmses_val, accs_test, f1s_test, rmses_test = [], [], [], [], [], []
for iteration in range(repitition_count):
    print("Iteration ", (iteration + 1), "------------------------------------")

    # Establish RvNN model:
    model = establish_model(vocabulary_size, hidden_dim, class_count)

    # Gradient descent:
    losses = []
    count_samples = 0
    for epoch in range(epoch_count):
        indexs = [i for i in range(len(y_train))]
        for i in indexs:
            loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], learning_rate)
            #print("iteration ", i)
            losses.append(np.round(loss,2))
            count_samples += 1
        print("Epoch: ", epoch, " Loss: ", np.mean(losses))
        
        if epoch % 5 == 0:
            prediction_dev, prediction_test = [], []
            for j in range(len(y_dev)):
                prediction_dev.append(model.predict_up(word_dev[j], index_dev[j], parent_num_dev[j], tree_dev[j]))
            print("Validation:")
            acc_val, f1_val, rmse_val = evaluate(y_dev, prediction_dev)
            for j in range(len(y_test)):
                prediction_test.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j]))
            print("Test:")
            acc_test, f1_test, rmse_test = evaluate(y_test, prediction_test)
        losses = []
    accs_val.append(acc_val)
    accs_test.append(acc_test)
    f1s_val.append(f1_val)
    f1s_test.append(f1_test)
    rmses_val.append(rmse_val)
    rmses_test.append(rmse_test)

print("Final results:")
for i in range(len(accs_val)):
    print("Iteration ", (i + 1), "------------------------------------")
    print("Validation: Accuracy ", accs_val[i], " F1: ", f1s_val[i], " RMSE: ", rmses_val)
    print("Test: Accuracy ", accs_test[i], " F1: ", f1s_test[i], " RMSE: ", rmses_test)
