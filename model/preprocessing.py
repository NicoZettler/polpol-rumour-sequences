from typing import Dict
import json

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
stop_words.extend([":", ",", ".", "#", "@", "-", "(", ")", ";", "&", "'"])
from nltk.stem import PorterStemmer
ps = PorterStemmer()

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
                
# removes unimportant constructs, tokenizes and stems a (str) post and turns it into a list of (str) words
def preprocess_words(raw_post: str) -> list:
    post_words = []
    for word in raw_post.split():
        if len(post_words) < 20: # make sure that posts don't contain more than 20 words (clip bigger posts)
            if not word.startswith('@') and not word.startswith('http'): # remove mentions and URLs
                post_words.append(word)
            elif word.startswith('http'):
                post_words.append('http') # for each link put http such that URLs can be identified for the indexing
    clean_post = ' '.join(post_words) # put post back together
    text_information = word_tokenize(clean_post) # 
    text_information = [ps.stem(word) for word in text_information] # stem all words
    return text_information
    
def calc_word_index_freq_pairs(text_information: list, word_index: int, maxL: int, words: Dict) -> (str, int, int):
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
    
# function for calculating the dictionary containing all the tree information for the RvNN
def calc_treeDic(treeDic: Dict, thread: Dict, twitter: bool, word_index: int,
                highest_source_eid: int, archive: Dict, indexDic: Dict, words: Dict) -> (Dict, int, int):
    maxL = 0 # maximum post length of each thread => reset to 0 for each new thread
    # get the source information as a Dict (contains all info about the source post)
    # somehow the folder is also named 'source-tweet' in the reddit data
    source_information = json.loads(archive.read(list(thread['source-tweet'].values())[0]))
    if twitter:
        idx = source_information['id'] # get the 18 digit source ID
    else: # get reddit ID
        idx = source_information['data']['children'][0]['data']['id']
    eid = indexDic[str(idx)] # convert it to the corresponding simpler ID
    post_structure = json.loads(archive.read(thread['structure.json'])) # get the thread structure as a Dict
    parent_num = calc_parent_num(post_structure) # calculate the number of reply levels in each thread structure
    indexC = eid # initialize post index with source post index

    # preprocessing of current source post
    if twitter:
        text_information = preprocess_words(source_information['text'])
    else: # reddit
        text_information = preprocess_words(source_information['data']['children'][0]['data']['title'])
    #print(text_information)

    Vec, word_index, maxL = calc_word_index_freq_pairs(text_information, word_index, maxL, words)
    #print(Vec, " wi: ", word_index, " maxL: ", maxL)

    if idx not in treeDic: # create empty entry first to make the key accessable
        treeDic[idx] = {}
    treeDic[idx][indexC] = {'parent':'None', 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
    if 'replies' in thread: # some "replies" folders seem to be empty and then this for loop would throw an error
        for reply in thread['replies'].values(): # for every reply post
            reply_empty = True # reset this value for every reply to check if reddit replies are empty
            # get the reply information as a Dict (contains all info about the reply post)
            reply_information = json.loads(archive.read(reply))
            if not twitter:
                if 'body' in reply_information['data']:
                    reply_empty = False
            if twitter or not reply_empty:
                if twitter:
                    reply_idx = reply_information['id']
                else: # reddit
                    reply_idx = reply_information['data']['id']
                indexC = highest_source_eid # convert 18 digit index to simpler index
                indexDic[str(reply_idx)] = highest_source_eid # save connection between IDs (18 digit and simple)
                # find out parent of each reply node
                indexP = find_parent_node(post_structure, str(reply_idx))

                # preprocessing of current reply post
                if twitter:
                    text_information = preprocess_words(reply_information['text'])
                else: # reddit
                    text_information = preprocess_words(reply_information['data']['body'])

                Vec, word_index, maxL = calc_word_index_freq_pairs(text_information, word_index, maxL, words)

                treeDic[idx][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
                highest_source_eid += 1 # increase index for the next reply
    for post in treeDic[idx].values(): # go through all posts again to set the maxL for every thread
        post['maxL'] = maxL
    return treeDic, word_index, highest_source_eid

def trim_tree(treeDic: Dict, indexDic: Dict) -> Dict:
    for i,v in treeDic.items():
        for j,w in list(v.items()):
            if not w['parent'] == 'None':
                if w['parent'] in indexDic:
                    w['parent'] = indexDic[w['parent']]
                else: # delete all branches in the tree which start with IDs that don't appear in the actual data 
                    print(w['parent'], w) # (because there is no statistical significance if we can't track the posts
                    print("whole branch ", treeDic[i][j])   # they refer to)
                    del treeDic[i][j]
                if w['vec'] == '':
                    #print(w['parent'], w)
                    print("vec whole branch ", treeDic[i][j])
                    del treeDic[i][j]
    return treeDic

def fit_to_model(word_data: list, index_data: list, tree_data: list) -> (list, list, list):
    # adapt word_data (pad zeros such that numpy does not throw any errors)
    for j in range(0,len(word_data)):
        word_ext = []
        for i in range(0,10000):
            if i < len(word_data[j]):
                word_ext.append(word_data[j][i])
            else:
                word_ext.append([0] * len(word_data[j][0])) # pad lines of zeros
        word_data[j] = word_ext
    # and also index_data and tree_data (because some trees are too small to handle)
    for i in range(0,len(index_data)):
        index_ext = list(index_data[i])
        tree_ext = list(tree_data[i])
        while len(index_ext) < 10: # pad zeros because RvNN doesn't work with smaller arrays
            index_ext.append([0] * len(index_data[i][0]))
            tree_ext.append([0] * 2)
        index_data[i] = index_ext
        tree_data[i] = tree_ext
    return word_data, index_data, tree_data