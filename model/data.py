# Top-down RvNN implementation based on the model of Jing Ma et al.
# (https://github.com/majingCUHK/Rumor_RvNN ; state: 10.09.2019)
# to improve the results of the rumor verification accomplished by the CLEARumor approach by Ipek Baris et al.
# (https://github.com/Institute-Web-Science-and-Technologies/CLEARumor ; state: 10.09.2019).
# @authors: Dhurim Sylejmani and Nico Zettler

from zipfile import ZipFile
from typing import Dict
from pathlib import Path
import json

# Paths for RumourEval-2019 training and test data
resource_path = Path('resource')
train_path = resource_path / "rumoureval-2019-training-data.zip"
test_path = resource_path / "rumoureval-2019-test-data.zip"
test_label_path = resource_path / "final-eval-key.json" # Evaluation file contains test labels

def load_data() -> (Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict, ZipFile, ZipFile):
    # Function to parse .zip files based on the implementation by Ipek et al.
    # (https://github.com/Institute-Web-Science-and-Technologies/CLEARumor ; state: 10.09.2019)
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
        
    training_data_archive = ZipFile(train_path)
    test_data_archive = ZipFile(test_path)
    training_data_contents = get_archive_directory_structure(
            training_data_archive)
    test_data_contents = get_archive_directory_structure(test_data_archive)
    train_data = json.loads(training_data_archive.read(training_data_contents['train-key.json']))
    dev_data = json.loads(training_data_archive.read(training_data_contents['dev-key.json']))
    test_data = json.load(open(test_label_path,'r'))
    twitter_train_data = training_data_contents['twitter-english']
    twitter_test_data = test_data_contents['twitter-en-test-data']
    reddit_train_data = training_data_contents['reddit-training-data']
    reddit_dev_data = training_data_contents['reddit-dev-data']
    reddit_test_data = test_data_contents['reddit-test-data']
    
    return train_data, dev_data, test_data, twitter_train_data, twitter_test_data, \
        reddit_train_data, reddit_dev_data, reddit_test_data, training_data_archive, \
        test_data_archive

# Loads all labels (train/dev/test) into one dictionary as (sourceID:label)
# and all IDs for training, dev and test data into three separate lists.
def load_labels(train_data: Dict, dev_data: Dict, test_data: Dict) -> (Dict, list, list, list):
    labelDic = {} # labelDic contains all (eid,label) connections.
    train_IDs, dev_IDs, test_IDs = [], [], []
    for (idx, label) in train_data['subtaskbenglish'].items():
        labelDic[idx] = label.lower()
        train_IDs.append(idx)
    for (idx, label) in dev_data['subtaskbenglish'].items():
        labelDic[idx] = label.lower()
        dev_IDs.append(idx)
    for (idx, label) in test_data['subtaskbenglish'].items():
        labelDic[idx] = label.lower()
        test_IDs.append(idx)
    return labelDic, train_IDs, dev_IDs, test_IDs
