#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way", "Dan Malmer", "Will Van Treuren", "Rob Knight"]
__license__ = "GPL"
__version__ = "1.7.0-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Scripts for evaluating scikit-learn models 
"""

from .cross_validation import get_test_train_set
from time import clock 

def get_evaluation_report(models, data_matrix, labels, test_sets, metrics)
    """ Get prediction results for a given list of models, using the supplied test sets 
        Inputs:
            models       : list of tuples, each tuple containing a name and model 
                           (must implement .predict() and .fit())
            data_matrix  : matrix of features for each example
            labels       : actual value for each example
            test_sets    : tuples of indicies for training and test sets
            metrics      : list of tuples, each containing a name and a metric for evaluation 
    """

    evaluation_report = {} 
    for model_name, model in models:
        # For each model, measure train/run time and all supplied metrics
        evaluation_report[model_name] = {'train_time':[],'run_time':[]}
        for metric_name, metric in metrics:
            evaluation_report[model_name][metric_name] = [] 
        
    for test_set in test_sets:

        training_matrix, training_labels, test_matrix, test_labels = \
            get_test_train_set(data_matrix, labels, test_set)

        for model_name, model in models:

            # Training 
            start_time = clock() 
            model.fit(training_matrix, training_labels)
            stop_time = clock() 
            evaluation_report[model_name]['train_time'].append(stop_time-start_time)

            # Testing 
            start_time = clock()
            predictions = model.predict(test_matrix)
            stop_time = clock() 
            evaluation_report[model_name]['run_time'].append(stop_time-start_time)

            # Evaluation 
            for metric_name, metric in metrics: 
                evaluation_report[model_name][metric_name].append(metric(test_labels, predictions))
            
    return evaluation_report 

