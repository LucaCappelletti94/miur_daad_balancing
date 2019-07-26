from miur_daad_balancing import full_balanced, load_full_balanced, load_balanced
import numpy as np
from .utils import sample_data, compare_tuples, sampling_class_portion, truncate_sample_size

def test_umbalanced():
    training, testing = sample_data()
    X_train, y_train, _, _ = truncate_sample_size(*training, max_size_given=load_balanced()["max"])
    balanced_training = (X_train, y_train)
    X_test, y_test, _, _ = sampling_class_portion(*testing, class_portion=load_full_balanced())
    balanced_testing = (X_test, y_test)
    #assert compare_tuples(training, balanced_training)
    #assert compare_tuples(testing, balanced_testing)