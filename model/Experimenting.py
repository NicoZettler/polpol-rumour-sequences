import TD_RvNN

obj = "Twitter15" # choose dataset, you can choose either "Twitter15" or "Twitter16"

#--------------LOAD LABELS--------------------------------------
labelPath = "../resource/"+obj+"_label_All.txt"
labelDic = {}
for line in open(labelPath):
    line = line.rstrip()
    label, eid = line.split('\t')[0], line.split('\t')[2]
    labelDic[eid] = label.lower()
#--------------LOAD LABELS--------------------------------------
#--------------LOAD TREE--------------------------------------
vocabulary_size = 5000
treePath = '../resource/data.TD_RvNN.vol_'+str(vocabulary_size)+'.txt'
treeDic = {}
for line in open(treePath):
    line = line.rstrip()
    eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
    parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])  
    Vec =  line.split('\t')[5] 
    #if not treeDic.has_key(eid):
    if eid not in treeDic:
        treeDic[eid] = {}
    treeDic[eid][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':maxL, 'vec':Vec}
print('tree no:', len(treeDic))
#print(treeDic)
#--------------LOAD TREE--------------------------------------
#--------------LOAD LABEL FUNCTION AND REQUIREMENTS--------------------------------------
def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
       y_train = [1,0,0,0]
       l1 += 1
    if label in labelset_f:
       y_train = [0,1,0,0] 
       l2 += 1
    if label in labelset_t:
       y_train = [0,0,1,0] 
       l3 += 1 
    if label in labelset_u:
       y_train = [0,0,0,1] 
       l4 += 1
    return y_train, l1,l2,l3,l4
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
        #print tree[j]['maxL']
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        #nodeC.time = tree[j]['post_t']
        ## not root node ## 
        if not indexP == 'None':
           nodeP = index2node[int(indexP)]
           nodeC.parent = nodeP
           nodeP.children.append(nodeC)
        ## root node ##
        else:
           root = nodeC
    ## 3. convert tree to DNN input    
    parent_num = tree[j]['parent_num'] 
    ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
    #x_word, x_index, tree = tree_gru_u2b.gen_nn_inputs(root, ini_x, ini_index) 
    x_word, x_index, tree = TD_RvNN.gen_nn_inputs(root, ini_x) 
    return x_word, x_index, tree, parent_num
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
    #print MaxL, l, len(Str.split(' ')), len(wordFreq)
    #print Str.split(' ')
    return wordFreq, wordIndex 
fold = "2" # fold index, choose from 0-4
trainPath = "../nfold/RNNtrainSet_"+obj+str(fold)+"_tree.txt" 
testPath = "../nfold/RNNtestSet_"+obj+str(fold)+"_tree.txt"
#--------------LOAD LABEL FUNCTION AND REQUIREMENTS--------------------------------------
#--------------LOAD TRAINING SET------------------------------
print("loading train set",)
tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
l1,l2,l3,l4 = 0,0,0,0
for eid in open(trainPath):
    #if c > 8: break
    eid = eid.rstrip()
    #if not labelDic.has_key(eid): continue
    #if not treeDic.has_key(eid): continue 
    if eid not in labelDic: continue
    if eid not in treeDic: continue
    if len(treeDic[eid]) <= 0: 
       #print labelDic[eid]
       continue
    ## 1. load label
    label = labelDic[eid]
    y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
    y_train.append(y)
    ## 2. construct tree
    #print eid
    x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
    tree_train.append(tree)
    word_train.append(x_word)
    index_train.append(x_index)
    parent_num_train.append(parent_num)
    #print treeDic[eid]
    #print tree, child_num
    #exit(0)
    c += 1
print(l1,l2,l3,l4)
#--------------LOAD TRAINING SET------------------------------
#--------------LOAD TEST SET------------------------------
print("loading test set",)
tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
l1,l2,l3,l4 = 0,0,0,0
for eid in open(testPath):
    #if c > 4: break
    eid = eid.rstrip()
    #if not labelDic.has_key(eid): continue
    #if not treeDic.has_key(eid): continue 
    if eid not in labelDic: continue
    if eid not in treeDic: continue
    if len(treeDic[eid]) <= 0: 
       #print labelDic[eid] 
       continue        
    ## 1. load label        
    label = labelDic[eid]
    y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
    y_test.append(y)
    ## 2. construct tree
    x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
    tree_test.append(tree)
    word_test.append(x_word)  
    index_test.append(x_index) 
    parent_num_test.append(parent_num)
    c += 1
print(l1,l2,l3,l4)
#--------------LOAD TEST SET------------------------------
print("Everything is working up to here!")