#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.7.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Scripts for performing cross validation
"""

from sklearn.cross_validation import StratifiedKFold, KFold
from numpy import array

def get_test_sets(class_labels, kfold=10, stratified=True):
    """ Generate lists of indices for performing k-fold cross validation
        Note: test sets are stratefied, meaning that the number of class labels
        in each test set is proportional to the number in the whole set. 
    """
    if stratified: 
        return StratifiedKFold(class_labels, kfold)
    else: 
        return KFold(len(class_labels), kfold)

def save_test_sets_to_file(test_sets, output_file):
    """ Save train/test indices to a file 
        Each pair of lines denotes the training and testing 
        indices for a single train-test set 
    """ 
    output = open(output_file, 'w')
    for train_idx, test_idx in test_sets:
        output.write(','.join([ str(i) for i in train_idx ]) + '\n')
        output.write(','.join([ str(i) for i in test_idx ]) + '\n')
    output.close()

def load_test_sets_from_file(input_file):
    """ Read in test sets from file """ 
    test_sets = [] 
    temp_set = {'train':None, 'test':None}
    currently_parsing = 'train' # start by parsing training

    for line in open(input_file, 'rU'):
        temp_set[currently_parsing] = array([int(x) for x in line.split(',')])
        if currently_parsing == 'test': 
            test_sets.append( (temp_set['train'], temp_set['test']) )
            currently_parsing = 'train'
        else:
            currently_parsing = 'test'
    return test_sets

def get_test_train_set(data_matrix, actual_values, test_set, is_distance_matrix=False):
    """ Create test/train data set for cross-validation 
        Inputs:
            + "data_matrix": sample/feature matrix 
            + "actual_values": labels for each of the samples in data_matrix    
            + "test_set": tuple of index lists for training and testing examples 
            + "is_distance_matrix": data_matrix is a distance matrix. 
    """
    train_idx, test_idx = test_set

    if not is_distance_matrix:
        train_matrix = data_matrix[train_idx,:]
        train_values = actual_values[train_idx]
        test_matrix = data_matrix[test_idx,:]
        test_values = actual_values[test_idx]
    else:
        train_matrix = data_matrix[train_idx,:][:,train_idx]
        train_values = actual_values[train_idx]
        test_matrix = data_matrix[test_idx,:][:,train_idx]
        test_values = actual_values[test_idx]

    return train_matrix, train_values, test_matrix, test_values 

