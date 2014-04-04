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

def is_iterable(item):
    """ Check if item is list-like """ 
    return isinstance(item, (list, tuple)) and \
        not isinstance(item, basestring)

def convert_labels_to_int(labels):
    """ Convert a list of labels to indices """ 
    if not len(labels):
        raise ValueError("Nothing to convert!")

    if isinstance(labels[0], basestring):
        label_legend = list(set(labels))
        converted_labels = [ label_legend.index(l) for l in labels ]
    else:  # Multiple labels, handle each one individually
        num_labels = len(labels[0])
        label_legend = []
        converted_labels = []
    
        # Get the unique sets of labels for each index
        for i in xrange(num_labels):
            temp_labels = [l[i] for l in labels]
            label_legend.append(list(set(temp_labels)))

        # Apply mapping to each label
        for label in labels:
            converted_label = [ leg.index(l) for leg, l in 
                                zip(label_legend, label) ]
            converted_labels.append(converted_label)

    return label_legend, array(converted_labels)

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
