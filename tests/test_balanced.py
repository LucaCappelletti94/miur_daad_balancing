from miur_daad_balancing import balanced, load_balanced
import numpy as np
from .utils import sample_data, compare_tuples, truncate_sample_size

def test_umbalanced():
    training, testing = sample_data()
    X_train, y_train, _, _ = truncate_sample_size(*training, max_size_given=load_balanced()["max"])
    balanced_training = (X_train, y_train)
    #assert compare_tuples(training, balanced_training)
    #assert compare_tuples(testing, balanced_testing)