from typing import Dict
import model.model
    
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

# construct tree function
def construct_tree(tree):
    index2node = {}
    for i in tree:
        node = model.model.Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        if not indexP == 'None': # not root node
            if int(indexP) in index2node: # there are two IDs listed in the structure files which don't exist in the training set
                nodeP = index2node[int(indexP)]       # => filter them out
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
        else: # root node
            root = nodeC
    parent_num = tree[j]['parent_num'] 
    ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
    x_word, x_index, tree = model.model.gen_nn_inputs(root, ini_x) # put root node (with all children) and 0 array with maxL
    return x_word, x_index, tree, parent_num
    
def load_tree_data(indexDic: Dict, labelDic: Dict, treeDic: Dict, IDList: list) -> (list, list, list, list, list):
    # load training data
    tree_data, word_data, index_data, y_data, parent_num_data = [], [], [], [], []
    for eid in IDList:
        if indexDic[eid] not in labelDic: continue
        if len(eid) == 18:
            eid = int(eid) # twitter IDs have to be cast to int for key checking
        if eid not in treeDic:
            continue
        if len(treeDic[eid]) <= 0:
            continue
        label = labelDic[indexDic[str(eid)]]
        if label == 'false':
            y_data.append([1,0,0])
        elif label == 'true':
            y_data.append([0,1,0])
        else: # 'unverified'
            y_data.append([0,0,1])
        x_word, x_index, tree, parent_num = construct_tree(treeDic[eid])
        tree_data.append(tree)
        word_data.append(x_word)
        index_data.append(x_index)
        parent_num_data.append(parent_num)
    return tree_data, word_data, index_data, y_data, parent_num_data
