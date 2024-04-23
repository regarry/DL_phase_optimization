# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image
import argparse
from data_utils import generate_batch, complex_to_tensor

def gen_data():
    # set random seed for repeatability
    torch.manual_seed(99)
    np.random.seed(50)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    path_train = 'traininglocations/'
    if not (os.path.isdir(path_train)):
        os.mkdir(path_train)
        
    # print status
    print('=' * 50)
    print('Sampling examples for training')
    print('=' * 50)
    
    # batch size for generating training examples:
    # locations for phase mask learning are saved in batches of 16 for convenience
    ntrain = 10000
    nvalid = 1000
    batch_size_gen = 2
    # calculate the number of training batches to sample
    ntrain_batches = int(ntrain/ batch_size_gen)
    
    # generate examples for training
    labels_dict = {}
    for i in range(ntrain_batches):
        # sample a training example
        xyz, Nphotons = generate_batch(batch_size_gen)
        labels_dict[str(i)] = {'xyz':xyz, 'N': Nphotons}
        # print number of example
        print('Training Example [%d / %d]' % (i + 1, ntrain_batches))
        
    
    nvalid_batches = int(nvalid/ batch_size_gen)
    for i in range(nvalid_batches):
        xyz, Nphotons = generate_batch(batch_size_gen)
        labels_dict[str(i+ntrain_batches)] = {'xyz':xyz, 'N': Nphotons}
        print('Validation Example [%d / %d]' % (i + 1, nvalid_batches))
    
    path_labels = path_train + 'labels.pickle'
    with open(path_labels, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    
    gen_data()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    