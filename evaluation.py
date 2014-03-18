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

from cross_validation import get_test_train_set
from numpy import array
from time import clock 

def format_model_predictions(model_predictions, metrics):
    """ Takes in a list of tuples (model_name, predictions, true_values) and returns an 
        evaluation report.  Model results are grouped by model name, so entries can be
        supplied for a model, if desired. 

        Example usage: 
        model_predictions = [ ('svm', [1, 1, 1], [1, 1, 0]), ('rf', [1, 1, 1], [1, 1, 1]) ] 
        metrics = [('Accuracy', metrics.accuracy_score)]
        evaluation_report = format_model_predictions(model_predictions, metrics)
        print_evaluation_report(evaluation_report, 'report.txt')

    """ 
    evaluation_report = {}
    for model_name, predictions, true_values in model_predictions:
        evaluation_report[model_name] = { metric_name:[] for metric_name, metric in metrics}

    for model_name, predictions, true_values in model_predictions:
        for metric_name, metric in metrics:
            evaluation_report[model_name][metric_name].append(metric(true_values, predictions))

    return evaluation_report 

def get_evaluation_report(models, data_matrix, labels, test_sets, metrics):
    """ Get prediction results for a given list of models, using the supplied test sets 
        Includes timing information and whatever evaluation metrics you supply. 

        Inputs:
            models       : list of tuples, each tuple containing a name and model 
                           (must implement .predict() and .fit())
            data_matrix  : matrix of features for each example
            labels       : actual value for each example
            test_sets    : tuples of indicies for training and test sets
            metrics      : list of tuples, each containing a name and a metric for evaluation 
    """

    evaluation_report = {} 
    train_time_key = 'Training time (s)'
    run_time_key = 'Running time (s)'

    for model_name, model in models:
        # For each model, measure train/run time and all supplied metrics
        evaluation_report[model_name] = {train_time_key:[],run_time_key:[]}
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
            evaluation_report[model_name][train_time_key].append(stop_time-start_time)

            # Testing 
            start_time = clock()
            predictions = model.predict(test_matrix)
            stop_time = clock() 
            evaluation_report[model_name][run_time_key].append(stop_time-start_time)

            # Evaluation 
            for metric_name, metric in metrics: 
                evaluation_report[model_name][metric_name].append(metric(test_labels, predictions))
            
    return evaluation_report 

def print_evaluation_report(evaluation_report, output_file):
    """ Write an evaluation report out to a text file """ 
    model_names = evaluation_report.keys()
    metrics = evaluation_report[model_names[-1]].keys()
    longest_metric = max([len(metric) for metric in metrics])

    output = open(output_file, 'w')
    pad = longest_metric - len('Models')
    output.write('Models%s :\t' % (' '*pad))
    output.write('\t'.join(model_names))

    for metric in metrics:
        pad = longest_metric - len(metric)
        output.write('\n%s%s :' % (metric, ' '*pad))
        
        for model_name in model_names:
            temp = array(evaluation_report[model_name][metric])
            output.write('\t%.3f (+-%.2f)' % (temp.mean(), temp.std()))

    output.write('\n\n')
    output.close()     

def make_evaluation_report(models, data_matrix, labels, test_sets, metrics, output_file):
    """ Run and print evaluation report """ 
    evaluation_report = get_evaluation_report(models, data_matrix, labels, test_sets, metrics)
    print_evaluation_report(evaluation_report, output_file)

