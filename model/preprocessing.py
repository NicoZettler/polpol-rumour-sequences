# Top-down RvNN implementation based on the model of Jing Ma et al.
# (https://github.com/majingCUHK/Rumor_RvNN ; state: 10.09.2019)
# to improve the results of the rumor verification accomplished by the CLEARumor approach by Ipek Baris et al.
# (https://github.com/Institute-Web-Science-and-Technologies/CLEARumor ; state: 10.09.2019).
# @authors: Dhurim Sylejmani and Nico Zettler

from typing import Dict
import json

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
#stop_words.extend([":", ",", ".", "#", "@", "-", "(", ")", ";", "&", "'", "?", "!", "...", "[", "]"])
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Removes unimportant constructs, tokenizes and stems a (str) post and turns it into a list of (str) words.
def preprocess_words(raw_post: str) -> list:
    post_words = []
    for word in raw_post.split():
        # Make sure that posts don't contain more than 20 words (clip bigger posts).
        if len(post_words) < 20:
            if not word.startswith('http') and not 'www' in word: # Remove mentions and URLs.
                post_words.append(word.lower()) # Change everything to lower case letters.
            elif word.startswith('http') or 'www' in word:
                post_words.append('http') # For each link put http such that URLs can be identified for the indexing.
    clean_post = ' '.join(post_words) # Put post back together.
    text_information = word_tokenize(clean_post)
    text_information = [ps.stem(word) for word in text_information] # Stem all words.
    # Remove stop_words.
    for word in text_information:
        if word in stop_words:
            text_information.remove(word)
    return text_information
    
# Function that counts up the frequencies of words in a post and adds them to the vocabulary.
def count_frequencies(text_information: list, words: Dict) -> Dict:
    for word in text_information:
        if word not in words.keys():
            words[word] = 1 # Word hasn't been in the Dict yet => add it with frequency 1.
        else:
            words[word] += 1 # Count up the word frequency.
    return words
    
# Function for going recursively through nested reddit replies.
def handle_nested_reddit_replies(nested_replies: list, words: Dict) -> Dict:
    for child in nested_replies:
        if 'body' in child['data']:
            text_information = preprocess_words(child['data']['body'])
            words = count_frequencies(text_information, words)
            if not child['data']['replies'] == '': # More nested replies.
                # Go through them recursively.
                words = handle_nested_reddit_replies(child['data']['replies']['data']['children'], words)
    return words
    
# Iterates through all posts (Twitter and Reddit) and counts for each word how often it appears.
# Returns a dictionary with all words and their frequencies. 
def estimate_word_frequencies(archive: Dict, thread: Dict, words: Dict, twitter: bool) -> Dict:
    # Get the source information as a Dict (contains all info about the source post).
    # Somehow the folder is also named 'source-tweet' in the reddit data.
    source_information = json.loads(archive.read(list(thread['source-tweet'].values())[0]))
    # Preprocessing of current source post:
    if twitter:
        text_information = preprocess_words(source_information['text'])
    else: # Reddit
        full_post = source_information['data']['children'][0]['data']['title']
        text_information = preprocess_words(full_post)
    words = count_frequencies(text_information, words)
    # Go through reply posts:
    if 'replies' in thread: # Some "replies" folders seem to be empty and then this for loop would throw an error.
        for reply in thread['replies'].values(): # For every reply post:
            reply_empty = True # Reset this value to check if reddit replies are empty.
            # Get the reply information as a Dict (contains all info about the reply post):
            reply_information = json.loads(archive.read(reply))
            if not twitter:
                if 'body' in reply_information['data']:
                    reply_empty = False
            if twitter or not reply_empty:
                # Preprocessing of current reply post:
                if twitter:
                    text_information = preprocess_words(reply_information['text'])
                    words = count_frequencies(text_information, words)
                else: # Reddit
                    text_information = preprocess_words(reply_information['data']['body'])
                    words = count_frequencies(text_information, words)
                    # Go through nested reddit replies:
                    if not reply_information['data']['replies'] == '': # No nested replies.
                        nested_replies = reply_information['data']['replies']['data']['children'] # List of all children.
                        words = handle_nested_reddit_replies(nested_replies, words)
    return words

# Calculates parent_num (the depth of the post tree).
def calc_parent_num(tree_branch: Dict) -> int:
    if isinstance(tree_branch, Dict): # Go recursively through the tree.
        return 1 + (max(map(calc_parent_num, tree_branch.values())) if tree_branch else 0)
    return 0
    
# Calculates indexP (the parent index of a node).
def find_parent_node(tree_branch: Dict, reply_idx: int) -> int:
    for indexP, sub_branch in tree_branch.items(): # Keep indexP and go through the keys.
        if reply_idx in sub_branch: # Search for reply_idx in the keys.
            return indexP # If it is found, return the parent node.
        elif isinstance(sub_branch, dict): # Else check if the keys itself are a dictionary.
            parent_node = find_parent_node(sub_branch, reply_idx) # If so, recursively call again the function with the subtree.
            if parent_node is not None: # Return the value only if it exists.
                return parent_node
           
# Calculates for all words of the posts index and frequency pairs as well as the maximum post length for each post tree.
def estimate_word_index_freq_pairs(text_information: list, word_index: int, maxL: int, sorted_words: list) -> (str, int, int):
    Vec = str()
    words_per_post = {} # Dict with frequencies for every word in a post; reset for every post.
    count_post_length = 0 # Reset for every loop iteration.
    for word in text_information: # Iterate through words of post.      
        if word in sorted_words:
            count_post_length += 1
            if word not in words_per_post.keys():
                words_per_post[word] = 1
            else:
                words_per_post[word] += 1
            word_index = sorted_words.index(word)
    iteration = 0 # Count iterations to get the last iteration and not put ' ' at the end of Vec.
    for word in words_per_post.keys(): # Iterate through words of post a second time to get the right numbers.
        Vec += str(sorted_words.index(word)) + ':' + str(words_per_post[word])
        iteration += 1
        if iteration != len(words_per_post.keys()): # If it's not the end of the tweet.
            Vec += ' ' # Add space between the word index/frequency pairs.            
    if maxL < count_post_length: # New maximum post length found.
        maxL = count_post_length    
    return Vec, word_index, maxL

# Function for deleting replies that appear in the structure file, but don't appear
# in the "replies" folder => we have no data to classify them.
def delete_missing_replies(existing_replies: list, post_structure: Dict) -> Dict:
    for key, value in list(post_structure.items()):
        if str(key) + ".json" not in existing_replies:
            del post_structure[key]
        if isinstance(value, Dict):
            delete_missing_replies(existing_replies, value)
    return post_structure

# Some of the nested reddit replies can't be found in the "replies" folder,
# but within their parents' .json files => recursive function to extract their information.
def calc_tree_dic_nested_reddit_replies(tree_dic: Dict, parent_idx_connections: Dict, nested_replies: list,
                                       idx: int, eid: int, post_structure: Dict, word_index: int,
                                       sorted_words: list, maxL: int,
                                       check_again_later: bool, parent_num: int) -> (Dict, Dict, int, int, int, bool):
    for child in nested_replies:
        if 'body' in child['data']:
            reply_idx = child['data']['id']
            if reply_idx in parent_idx_connections.keys():
                return tree_dic, parent_idx_connections, eid, maxL, word_index, check_again_later
            # Find out parent of each nested reply.
            parent_idx = find_parent_node(post_structure, str(reply_idx))
            if parent_idx == None:
                return tree_dic, parent_idx_connections, eid, maxL, word_index, check_again_later
            else:
                eid += 1
                indexC = eid
            parent_idx_connections[reply_idx] = eid # keep track of ID
            if parent_idx in parent_idx_connections.keys():
                indexP = parent_idx_connections[parent_idx]
            else:
                check_again_later = True
            text_information = preprocess_words(child['data']['body'])
            Vec, word_index, maxL = estimate_word_index_freq_pairs(text_information, word_index, maxL, sorted_words)
            if Vec == '':
                eid -= 1 # Set ID one back, because the reply is not counted in the tree.
                return tree_dic, parent_idx_connections, eid, maxL, word_index, check_again_later
            if not check_again_later: # Parent was successfully found.
                tree_dic[idx][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
            else: # pPrent can't be found in first iteration through replies due to messed up post structure order.
                tree_dic[idx][indexC] = {'parent':0, 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
            if not child['data']['replies'] == '': # More nested replies.
                # Go through them recursively:
                tree_dic, parent_idx_connections, eid, maxL, word_index, check_again_later = \
                    calc_tree_dic_nested_reddit_replies(tree_dic, parent_idx_connections,
                                                       child['data']['replies']['data']['children'],
                                                       idx, eid, post_structure, word_index, sorted_words, maxL,
                                                       check_again_later, parent_num)
                                                       
    return tree_dic, parent_idx_connections, eid, maxL, word_index, check_again_later

# Function for calculating the dictionary containing all the tree information for the RvNN.
def calc_tree_dic(tree_dic: Dict, thread: Dict, twitter: bool, word_index: int,
                archive: Dict, sorted_words: list) -> (Dict, int):
    parent_idx_connections = {} # Dict to translate Twitter/Reddit ID to thread post ID.
    maxL = 0 # Maximum post length of each thread => reset to 0 for each new thread.
    # Get the source information as a Dict (contains all info about the source post).
    # Somehow the folder is also named 'source-tweet' in the reddit data.
    source_information = json.loads(archive.read(list(thread['source-tweet'].values())[0]))
    if twitter:
        idx = source_information['id'] # Get the 18 digit source ID.
    else: # Get reddit ID.
        idx = source_information['data']['children'][0]['data']['id']
    eid = 1 # Reset ID to 1 for each new thread.
    parent_idx_connections[idx] = eid # Keep track of ID.
    post_structure = json.loads(archive.read(thread['structure.json'])) # Get the thread structure as a Dict.
    if 'replies' in thread: # Some "replies" folders seem to be empty.
        if twitter: # In the twitter data some replies listed in the structure file don't appear in the "replies" folder.
            existing_replies = list(thread['replies'].keys()) # Get replies that actually exist in the "replies" folder.
            existing_replies.extend(list(thread['source-tweet'].keys())) # Add source post to not remove it accidentally.
            post_structure = delete_missing_replies(existing_replies, post_structure) # Filter all other replies out of post_structure.
        parent_num = calc_parent_num(post_structure) # Calculate the number of reply levels in each thread structure.
    else:
        parent_num = 1 # No reply folder => post_structure depth is 1.
    indexC = eid # Initialize post index with source post index.
    # Preprocessing of current source post:
    if twitter:
        text_information = preprocess_words(source_information['text'])
    else: # Reddit
        full_post = source_information['data']['children'][0]['data']['title']
        text_information = preprocess_words(full_post)        
    Vec, word_index, maxL = estimate_word_index_freq_pairs(text_information, word_index, maxL, sorted_words)
    if idx not in tree_dic: # Create empty entry first to make the key accessable.
        tree_dic[idx] = {}
    tree_dic[idx][indexC] = {'parent':'None', 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
    if 'replies' in thread: # Some "replies" folders seem to be empty and then this for loop would throw an error.
        check_again_later = False # Reset this value here.
        for reply in thread['replies'].values(): # For every reply post:
            reply_empty = True # Reset this value for every reply to check if reddit replies are empty.
            # Get the reply information as a Dict (contains all info about the reply post):
            reply_information = json.loads(archive.read(reply))
            if not twitter:
                if 'body' in reply_information['data']:
                    reply_empty = False
            if twitter or not reply_empty:
                if twitter:
                    reply_idx = reply_information['id']
                else: # Reddit
                    reply_idx = reply_information['data']['id']
                    if reply_idx in parent_idx_connections.keys():
                        continue
                # Find out parent of each reply node:
                parent_idx = find_parent_node(post_structure, str(reply_idx))
                if parent_idx == None:
                    continue
                else: # Only increase ID if reply actually has a parent.
                    eid += 1
                    indexC = eid
                parent_idx_connections[reply_idx] = eid # Keep track of ID.
                if twitter: # Needs to be casted to int for some reason.
                    if int(parent_idx) in parent_idx_connections.keys():
                        indexP = parent_idx_connections[int(parent_idx)]
                    else:
                        check_again_later = True # Some post structures are processed in the wrong order => iterate over them again later.
                else: # Reddit str
                    if parent_idx in parent_idx_connections.keys():
                        indexP = parent_idx_connections[parent_idx]
                    else:
                        check_again_later = True # Some post structures are processed in the wrong order => iterate over them again later.
                # Preprocessing of current reply post:
                if twitter:
                    text_information = preprocess_words(reply_information['text'])
                else: # Reddit
                    text_information = preprocess_words(reply_information['data']['body'])
                    # There is no selftext in reply posts.               
                Vec, word_index, maxL = estimate_word_index_freq_pairs(text_information, word_index, maxL, sorted_words)
                if Vec == '':
                    Vec = "4999:1" # Set to a trivial word.
                
                if not check_again_later: # Parent was successfully found.
                    tree_dic[idx][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':0, 'vec':Vec}
                else: # Parent can't be found in first iteration through replies due to messed up post structure order.
                    tree_dic[idx][indexC] = {'parent':0, 'parent_num':parent_num, 'maxL':0, 'vec':Vec}       
                # Reddit posts have even more replies within their reply information that also have to be putinto the tree structure
                if not twitter:
                    if not reply_information['data']['replies'] == '': # No nested replies.
                        nested_replies = reply_information['data']['replies']['data']['children'] # List of all children.
                        tree_dic, parent_idx_connections, eid, maxL, word_index, check_again_later = \
                            calc_tree_dic_nested_reddit_replies(tree_dic, parent_idx_connections, nested_replies,
                                                              idx, eid, post_structure, word_index, sorted_words, maxL,
                                                              check_again_later, parent_num)
                    
        if check_again_later:
            for reply in thread['replies'].values(): # Iterate through replies again now that all indices are stored in parent_idx_connections list.
                reply_empty = True # Reset this value for every reply to check if reddit replies are empty.
                reply_information = json.loads(archive.read(reply))
                if not twitter:
                    if 'body' in reply_information['data']:
                        reply_empty = False
                if twitter or not reply_empty:
                    if twitter:
                        reply_idx = reply_information['id']
                        if int(reply_idx) not in parent_idx_connections.keys():
                            continue
                    else: # Reddit
                        reply_idx = reply_information['data']['id']
                    indexC = parent_idx_connections[reply_idx]
                    # Do the same as above again:
                    parent_idx = find_parent_node(post_structure, str(reply_idx))
                    if twitter: # Needs to be casted to int for some reason.
                        if int(parent_idx) in parent_idx_connections.keys():
                            indexP = parent_idx_connections[int(parent_idx)]
                        else:
                            continue
                    else: # Reddit str
                        if parent_idx in parent_idx_connections.keys():
                            indexP = parent_idx_connections[parent_idx]
                        else:
                            continue
                    tree_dic[idx][indexC]['parent'] = indexP
                    if not twitter:
                        if not reply_information['data']['replies'] == '': # No nested replies.
                            nested_replies = reply_information['data']['replies']['data']['children'] # List of all children.
                            tree_dic, parent_idx_connections, eid, maxL, word_index, check_again_later = \
                                calc_tree_dic_nested_reddit_replies(tree_dic, parent_idx_connections, nested_replies,
                                                                  idx, eid, post_structure, word_index, sorted_words, maxL,
                                                                  False, parent_num)
    for post in tree_dic[idx].values(): # Go through all posts again to set the maxL for every thread.
        post['maxL'] = maxL
    return tree_dic, word_index
