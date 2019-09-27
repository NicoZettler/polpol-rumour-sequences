# Top-down RvNN implementation based on the model of Jing Ma et al.
# (https://github.com/majingCUHK/Rumor_RvNN ; state: 10.09.2019)
# to improve the results of the rumor verification accomplished by the CLEARumor approach by Ipek Baris et al.
# (https://github.com/Institute-Web-Science-and-Technologies/CLEARumor ; state: 10.09.2019).
# @authors: Dhurim Sylejmani and Nico Zettler

from typing import Dict
import model.model
    
# Function for splitting up 'vec' component (word frequencies and indices).
def split_vec(vec: str, maxL: int):
    word_freq, word_index = [], []
    l = 0
    for pair in vec.split(' '):
        word_freq.append(float(pair.split(':')[1]))
        word_index.append(int(pair.split(':')[0]))
        l += 1
    ladd = [0 for i in range(maxL - l)]
    word_freq += ladd 
    word_index += ladd 
    return word_freq, word_index

# Constructs the RvNN tree.
def construct_tree(tree):
    index2node = {}
    for i in tree:
        node = model.model.Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        word_freq, word_index = split_vec(tree[j]['vec'], tree[j]['maxL'])
        nodeC.index = word_index
        nodeC.word = word_freq
        if not indexP == 'None': # not root node
            if int(indexP) in index2node: # there are two IDs listed in the structure files which don't exist in the training set
                nodeP = index2node[int(indexP)]       # => filter them out
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
        else: # root node
            root = nodeC
    parent_num = tree[j]['parent_num'] 
    ini_x, ini_index = split_vec("0:0", tree[j]['maxL'])
    x_word, x_index, tree = model.model.gen_nn_inputs(root, ini_x) # put root node (with all children) and 0 array with maxL
    return x_word, x_index, tree, parent_num
    
# Loads training, dev and test data as tree structure.
def load_tree_data(label_dic: Dict, tree_dic: Dict, ID_list: list) -> (list, list, list, list, list):
    tree_data, word_data, index_data, y_data, parent_num_data = [], [], [], [], []
    for eid in ID_list:
        if str(eid) not in label_dic: continue
        if len(eid) == 18:
            eid = int(eid) # Twitter IDs have to be cast to int for key checking.
        if eid not in tree_dic:
            continue
        if len(tree_dic[eid]) <= 0:
            continue
        label = label_dic[str(eid)]
        if label == 'false':
            y_data.append([1,0,0])
        elif label == 'true':
            y_data.append([0,1,0])
        else: # 'unverified'
            y_data.append([0,0,1])
        x_word, x_index, tree, parent_num = construct_tree(tree_dic[eid])
        tree_data.append(tree)
        word_data.append(x_word)
        index_data.append(x_index)
        parent_num_data.append(parent_num)
    return tree_data, word_data, index_data, y_data, parent_num_data
