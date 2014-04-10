#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2011, The QIIME Project"
__credits__ = ["Sam Way"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

""" Scripts for evaluating scikit-learn models 
"""

from cross_validation import get_test_train_set
from util import is_iterable
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


def get_predictions(models, data_matrix, labels, test_sets):
    """ Get model predictions for the supplied test set as
        well as timing information for training and running
        each of the models. 

        INPUTS
        models - list of tuples where each tuple looks like
                 (model_name, model)
                 model should implement both fit() and predict()
                 methods in order to be tested here. 

        data_matrix - matrix containing feature values for
                      all samples.  Any necessary data
                      preprocessing should be performed 
                      prior to this step or included in the 
                      models.  
        labels - true labels/values that we wish to predict

        test_sets - list of tuples where each tuple looks like
                    (training indices, testing indices)
                    These lists of indicies will be used to 
                    slice out appropriate portions of the 
                    data matrix and labels.  

        OUTPUTS
        timing_info - dictionary indexed by model names
                      contains list of tuples where each tuple
                      is (train time, run time) for each 
                      of the test sets in test_sets. 

        model_predictions - dictionary indexed by model names
                            contains list of model predictions
                            for each of the test sets

        actual_values - list of actual labels/values corresponding 
                        to each of the test sets.
    """
    timing_info = {}
    model_predictions = {}
    actual_values = []

    for model_name, model in models:
        timing_info[model_name] = []
        model_predictions[model_name] = []
        
    for test_set in test_sets:

        training_matrix, training_labels, test_matrix, test_labels = \
            get_test_train_set(data_matrix, labels, test_set)
        actual_values.append(test_labels)

        for model_name, model in models:

            # Training 
            start_time = clock() 
            model.fit(training_matrix, training_labels)
            training_time = clock() - start_time

            # Testing 
            start_time = clock()
            predictions = model.predict(test_matrix)
            running_time = clock() - start_time

            model_predictions[model_name].append(predictions)
            timing_info[model_name].append((training_time, running_time))

    return timing_info, model_predictions, actual_values


def unpack_evaluations(model_predictions, actual_values):
    """ In the event that multiple outputs were predicted
        unpack them into separate results for easier
        shipping and handling """ 
    first_model = model_predictions.iterkeys().next()
    first_predictions = model_predictions[first_model][0]
    num_outputs = first_predictions.ndim

    if num_outputs == 1:
        return [(model_predictions, actual_values)]
    else:
        separate_model_predictions = []
        separate_actual_values = []

        for i in xrange(num_outputs):
            a_copy = [a[:,i] for a in actual_values]
            p_copy = model_predictions.copy()
            for model_name, predictions in p_copy.iteritems():
                p_copy[model_name] = \
                    [ p[:,i] for p in predictions ]
            separate_model_predictions.append(p_copy)
            separate_actual_values.append(a_copy)

        return zip(separate_model_predictions,
                   separate_actual_values)


def evaluate_models(models, data_matrix, labels, test_sets, 
                    metrics, output_handle):
    """ Get model predictions for the supplied test set as
        well as timing information for training and running
        each of the models. 

        INPUTS
        models - list of tuples where each tuple looks like
                 (model_name, model)
                 model should implement both fit() and predict()
                 methods in order to be tested here. 

        data_matrix - matrix containing feature values for
                      all samples.  Any necessary data
                      preprocessing should be performed 
                      prior to this step or included in the 
                      models.  
        labels - true labels/values that we wish to predict

        test_sets - list of tuples where each tuple looks like
                    (training indices, testing indices)
                    These lists of indicies will be used to 
                    slice out appropriate portions of the 
                    data matrix and labels.  

        metrics - list of tuples where each tuple contains
                  (metric_name, metric)
    """   
    model_names = [model_name for model_name, m in models]
    longest_name = max([len(m) for m in model_names])
    metric_names = [metric_name for metric_name, m in metrics]
    timing_info, model_predictions, actual_values = \
        get_predictions(models, data_matrix, labels, test_sets)
    output_number = 1

    unpacked = unpack_evaluations(model_predictions, actual_values)
    if len(unpacked) > 1:
        # Also show combined, if multiple outputs
        unpacked.append((model_predictions.copy(), actual_values[:]))

    for model_predictions, actual_values in unpacked: 
        if len(unpacked) > 1:
            if output_number == len(unpacked):
                output_handle.write('Combined Outputs\n')
            else:
                output_handle.write('Output: %d\n' % (output_number))
                output_number += 1
       
        output_handle.write(' '*(longest_name)
                            + '\tTraining (s)'
                            + '\tRunning (s)\t'
                            + '\t'.join(metric_names) 
                            + '\n')

        for model_name, model in models:
            pad = ' '*(longest_name-len(model_name))
            output_handle.write(pad + model_name + ': ')
            train_time = [t[0] for t in timing_info[model_name]]
            train_time = array(train_time).mean()
            run_time = [t[1] for t in timing_info[model_name]]
            run_time = array(run_time).mean()
            output_handle.write('\t%0.3f\t\t%0.3f\t' 
                % (train_time, run_time))

            for metric_name, metric in metrics:
                temp_values = []
                for true, pred in zip(actual_values,
                                      model_predictions[model_name]):
                    # tolist() conversion to avoid weird error
                    # ValueError: Can't handle mix of multiclass-multioutput and 
                    #             multilabel-indicator
                    # Line 115 of sklearn/metrics/metrics.py
                    temp_values.append(metric(true.tolist(), pred.tolist()))
                temp_values = array(temp_values)
                output_handle.write('\t%0.4f (+/- %0.4f)' % (temp_values.mean(), temp_values.std()*2))
            output_handle.write('\n')
        output_handle.write('\n')

'''
#Old code... What to do, what to do... 
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
'''
