from miur_daad_balancing import balanced, load_balanced
import numpy as np
from miur_daad_balancing.originals import truncate_sample_size, sampling_class_portion
from .utils import sample_data, compare_tuples

def test_balanced():
    training, testing = sample_data()
    np.random.seed(42)
    X_train, y_train = truncate_sample_size(training[:-1], training[-1], max_size_given=load_balanced()["max"])
    original_balanced_training = (*X_train, y_train)
    np.random.seed(42)
    balanced_training, balanced_testing = balanced(training, testing)
    assert compare_tuples(original_balanced_training, balanced_training)
    assert compare_tuples(testing, balanced_testing)