#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.8.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Machine Learning utility script 
"""

from numpy import array

def convert_labels_to_int(labels):
    """ Convert a list of labels to indices """ 
    unique_labels = list(set(labels))
    label_indices = array([ unique_labels.index(l) for l in labels ]) 
    return unique_labels, label_indices

def bool_cast(s):
    """ Cast string to boolean """
    if s.lower() == 'true' or s.lower() == 't':
        return True
    elif s.lower() == 'false' or s.lower() == 'f':
        return False
    raise ValueError('Could not cast to ')

def custom_cast(s):
    """ Convert to number/binary/string in that order of preference """
    for cast_func in (int, float, bool_cast, str):
        try:
            return cast_func(s)
        except ValueError:
            pass
    raise BaseException('Could not cast as number/string!')
