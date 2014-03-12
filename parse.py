#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.8.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Parsing functions for interfacing with biom tables and metadata files. 
"""

from numpy import asarray, array, zeros
from biom.parse import parse_biom_table
from biom.table import DenseTable
from qiime.parse import parse_mapping_file_to_dict, parse_distmat
from util import custom_cast
import pickle
import warnings

def save_predictions_to_file(predictions, filename):
    """ Save classification results to a pickle file """
    pickle.dump(predictions, open(filename, 'wb'))

def load_predictions_from_file(filename):
    """ Load classification/regression results from pickle file """
    return pickle.load(open(filename, 'rb'))

def load_dataset(data_matrix_file, mapping_file, metadata_category, \
    metadata_value, labels_file, is_distance_matrix):
    """ Parse and prepare data for processing. """

    if not is_distance_matrix:
        sample_ids, data_matrix = parse_otu_matrix(data_matrix_file)
    else:
        sample_ids, data_matrix = parse_distance_matrix(data_matrix_file)

    if mapping_file is not None:
        if metadata_category is None:
            print "To extract labels from a mapping file, you must supply the desired " + \
                  "metadata category!"
            exit()
        actual_values = parse_mapping_file_to_labels(mapping_file, sample_ids, \
            metadata_category, metadata_value)
    else:
        sample_ids = array([x.split('.')[0] for x in sample_ids]) # Hack to work with Dan's stuff
        label_dict =  parse_labels_file_to_dict(labels_file)
        data_matrix, sample_ids, actual_values = sync_labels_and_otu_matrix(data_matrix, \
            sample_ids, label_dict)

    return data_matrix, sample_ids, actual_values

def parse_otu_matrix(biom_file):
    """ Parses a (dense) OTU matrix from a biom file. 
        Outputs: Dense OTU matrix, list of sample ids
    """
    # Parse the OTU table into a dense matrix
    otu_table = parse_biom_table(open(biom_file,'U'))
    if isinstance(otu_table, DenseTable):
        otu_matrix = otu_table._data.T
    else:
        otu_matrix = asarray([v for v in otu_table.iterSampleData()])
    return array(otu_table.SampleIds), otu_matrix

def parse_distance_matrix(distance_matrix_file):
    """ Parses distance matrix file """
    sample_ids, distance_matrix = parse_distmat(open(distance_matrix_file, 'rU'))
    return sample_ids, distance_matrix

def get_metadata_categories_from_mapping_file(mapping_file):
    """ Print list of metadata categories """ 
    mapping_fp = open(mapping_file, 'rU')
    mapping_dict, comments = parse_mapping_file_to_dict(mapping_fp)
    sample_id = mapping_dict.keys()[0]
    return mapping_dict[key].keys()

def parse_metadata_category_from_mapping_file(mapping_file, metadata_category):
    """ Returns a dictionary mapping sample ids to values from a specified metadata category """ 
    mapping_fp = open(mapping_file, 'rU')
    mapping_dict, comments = parse_mapping_file_to_dict(mapping_fp)
    simple_dict = { key:mapping_dict[key][metadata_category] for key in mapping_dict.keys() } 
    return simple_dict

def parse_mapping_file_to_labels(mapping_file, sample_ids, metadata_category, metadata_value=None):
    """ Extracts the specified metadata category from the mapping file for each of the 
        sample ids in sample_ids.  Returns a boolean list of values if metadata_value is supplied """
    mapping_fp = open(mapping_file, 'rU')
    mapping_dict, comments = parse_mapping_file_to_dict(mapping_fp)

    class_labels = []
    for sample_id in sample_ids:
        try:
            class_labels.append(mapping_dict[sample_id][metadata_category])
        except KeyError:
            if sample_id not in mapping_dict.keys():
                warnings.warn('Mapping file missing sample id: %s' % sample_id)
            else:
                raise Exception('Mapping file missing category: %s' % metadata_category)
    if metadata_value is not None:
        class_labels = [ label==metadata_value for label in class_labels ]
        if True not in class_labels:
            raise ValueError('No samples have the specified metadata_value (%s)' % \
                (metadata_value))
    else:
        # If no value is supplied, prefer numeric metadata values
        class_labels = [ custom_cast(label) for label in class_labels ] 
    return array(class_labels)

def parse_labels_file_to_dict(labels_file):
    """ Parse id-label file to a dictionary """ 
    label_dict = {}
    for line in open(labels_file, 'rU'):
        pieces = line.strip().split('\t')
        if len(pieces) != 2: continue 
        label_dict[pieces[0]] = custom_cast(pieces[1])
    return label_dict

def sync_labels_and_otu_matrix(otu_matrix, sample_ids, labels_dict):
    """ Returns appropriate rows of an otu matrix and corresponding sample id vector 
        to include all sample ids in labels_dict """ 
    select = array([ i for i in xrange(len(otu_matrix)) if sample_ids[i] in labels_dict.keys()])
    otu_matrix = otu_matrix[select, :]
    sample_ids = sample_ids[select]
    class_labels = array([labels_dict[sid] for sid in sample_ids])
    return otu_matrix, sample_ids, class_labels

def parse_confusion_matrix_file(input_file, normalized=True):
    """ Parses the output of 'confusion_matrix.txt,' the file
        generated by supervised_learning.py's R script

        Assumes format:
        True\Predicted\t class_a\t class_b\t Class error
        class_a\t 197\t 40\t 0.168776371308017
        class_b\t 19\t 344\t 0.0523415977961433
    """ 
    input_fp = open(input_file, 'rU')
    first_line = input_fp.readline()
    line_pieces = first_line.split('\t')
    
    if line_pieces[0] != "True\Predicted":
        raise ValueError('Input file not in expected format\n' + \
            'First line should start with "True\Predicted."') 
        
    labels = line_pieces[1:-1]
    N = len(labels)
    confusion_matrix = zeros((N, N))
    for i in xrange(N):
        line_pieces = input_fp.readline().split('\t')
        values = array([ float(x) for x in line_pieces[1:-1] ])
        total = sum(values)
        if normalized: 
            if total > 0: 
                values = values / total
            else:
                raise ValueError('Row for "%s" contains all zeros' % (labels[i]))
        confusion_matrix[i] = values 
    return confusion_matrix, labels 

