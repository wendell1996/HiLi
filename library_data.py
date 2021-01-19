'''
This is a supporting library for the loading the data.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
Highly Liquid Temporal Interaction Graph Embeddings. Huidi Chen, Yun Xiong, yangyong Zhu, Philip S. Yu. The Web Conference, 2021.
'''

import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
from collections import OrderedDict
import os, re
import argparse
from sklearn.preprocessing import scale
import pickle as pkl

class WindowsDict(OrderedDict):
    '''
    frequent previous nodes windows 
    '''
    def __init__(self, capacity=3, default=None):
        super(WindowsDict,self).__init__()
        if capacity <= 0:
            raise ValueError("'Capacity' must be positive.")
        self._capacity = capacity
        self._size = 0 
        self._default = default
        if not default is None:
            self[default] = 0
            self._size = 1
    
    def push(self, item):
        if list(self.keys()) == [self._default]:
            self._size = 0
            self.clear()
        self._push(item)
    
    def _push(self, item):
        if self._capacity > self._size:
            self._size += 1
            self.setdefault(item,0)
            self[item] += 1
        else:
            fkey = list(self.keys())[0]
            if self[fkey] == 1:
                self.popitem(0)
            else:
                self[fkey] -= 1
            self.setdefault(item,0)
            self[item] += 1
            self.move_to_end(item)
        

# LOAD THE NETWORK
def load_network(args, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    network = args.dataset
    datapath = args.datapath
    dataroot = os.path.split(datapath)[0]
    if os.path.exists(os.path.join(dataroot,f'{args.dataset}.dat')):
        with open(os.path.join(dataroot,f'{args.dataset}.dat'),'rb') as f:
            res = pkl.load(f)
            return res

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (network,
        datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    res = [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels]
    
    with open(os.path.join(dataroot,f'{args.dataset}.dat'),'wb') as f:
        pkl.dump(res, f)
    return res

# LOAD THE NETWORK
def load_data(args, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    network = args.dataset
    datapath = args.datapath
    dataroot = os.path.split(datapath)[0]
    size = args.size

    if os.path.exists(os.path.join(dataroot,f'{args.dataset}_{args.size}_f.dat')):
        with open(os.path.join(dataroot,f'{args.dataset}_{args.size}_f.dat'),'rb') as f:
            res = pkl.load(f)
            return res

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (network,
        datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    user_frequence = []
    user_pre = defaultdict(set)
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: WindowsDict(size,default=num_items))
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user].copy())
        user_frequence.append(len(user_pre[user]))
        user_pre[user].add(item2id[item_sequence[cnt]])
        user_latest_itemid[user].push(item2id[item_sequence[cnt]])
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    res = [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels, user_frequence]
    
    with open(os.path.join(dataroot,f'{args.dataset}_{args.size}_f.dat'),'wb') as f:
        pkl.dump(res, f)
    return res

