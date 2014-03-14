#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.8.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Functions for interfacing with SLDA code
"""

def write_matrix_to_slda_file(data_matrix, output_file):
    """ Format matrix for input to SLDA """ 
    output = open(output_file, 'w')
    N = data_matrix.shape[1] 
    for row in data_matrix:
        to_write = [ '%d:%d'%(k, row[k]) for k in xrange(N) if row[k] > 0 ] 
        output.write('%d %s\n' % (len(to_write), ' '.join(to_write)))
    output.close()

