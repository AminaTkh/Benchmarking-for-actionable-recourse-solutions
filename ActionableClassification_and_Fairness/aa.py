import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
from gurobipy import *
from actionable_classification_algorithm import ActionableClassificationAlgorithm
from flipset_algorithm import FlipsetAlgorithm, UNACTIONABLE
from time import time


def actionable_classification(data,  # Data either a DataFrame or tuple (x, y)
                              costs,
                              clf=None,  # Will construct classifier if none given
                              clf_type='LR',
                              output_file=None,  # Will not print results if file not set
                              action_fairness=False,  # Flag of whether to run Action Fairness
                              sensitive_attr_index=None,
                              proportional_strictness=4,  # Proportional strictness for Action Fairness
                              number_of_blocks=20,  # How many blocks to create in Action Fairness
                              
                              ):
    """Calculate flipset of actions for negative outcomes to be classified as positive."""
    # Get input data
    if isinstance(data, pd.DataFrame):
        attr_names = data.columns.to_list()[1:]  # Get names of the attributes (apart from name of classification label)
        y = data.iloc[:, 0]  # Classification is first column
        x = data.iloc[:, 1:]
    elif isinstance(data, tuple):
        # Tuple is (x, y)
        x = data[0]
        y = data[1]
        attr_names = x.columns.to_list()
        assert len(x) == len(y), "x and y input data must be of the same length"
    else:
        raise TypeError("Input Data must be of type DataFrame or (x, y) where x is a DataFrame and y is a Series")

    _check_parameters(output_file, x.shape, costs, clf)
    if clf is None:
        clf = _make_classifier(x, y, clf_type)  # Create classifier if none given

    alg = ActionableClassificationAlgorithm(costs, clf, clf_type, output_file, x, y, attr_names,
                                            sensitive_attr_index, proportional_strictness, number_of_blocks, action_fairness
                                            )
    alg.run()
    return alg


def get_individual_flipset(clf, costs, inputs, min_max):
    """Calculate and return a flipset for an individual."""
    f = FlipsetAlgorithm(clf, costs, inputs, m=None, for_individual=True, min_max=min_max)
    f.run()
    return f.old_values, f.new_values


def _make_classifier(x, y, clf_type='LR'):
    """Return a linear classifier, trained on the csv data."""
    clf = None
    if clf_type.upper() == 'SVM':
        clf = SVC(kernel='linear', probability=True)
    elif clf_type.upper() == 'LR':
        clf = LogisticRegression(solver='liblinear')
    else:
        raise TypeError("Entered an invalid classifier type. Accepted types are 'SVM' and 'LR'")

    clf.fit(x, y)
    return clf


def _check_parameters(output_file, data_shape, costs, clf):
    """Perform checks on method parameters."""
    # Delete content of gurobi.log (if exists)
    open('gurobi.log', 'w+').close()

    # Delete output file (if it exists)
    if output_file is not None:
        try:
            os.remove(output_file)
            print("Deleted old file with the name \'" + output_file + "\'")
        except FileNotFoundError:
            print("There was no existing file with the name \'" + output_file + "\'")

    if isinstance(clf, sklearn.svm.SVC) and not clf.probability:
        print("SVM classifiers need to have 'probability=True'")

    # Check cost list is of the correct length
    assert data_shape[1] == len(costs), "List of costs is a different length to the number of attributes in input data"
