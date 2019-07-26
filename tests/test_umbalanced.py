from miur_daad_balancing import umbalanced
import numpy as np
from .utils import sample_data, compare_tuples

def test_umbalanced():
    training, testing = sample_data()
    balanced_training, balanced_testing = umbalanced(training, testing)
    assert compare_tuples(training, balanced_training)
    assert compare_tuples(testing, balanced_testing)