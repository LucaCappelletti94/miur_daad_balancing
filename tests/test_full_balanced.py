from miur_daad_balancing import full_balanced, load_balanced, load_full_balanced
import numpy as np
from miur_daad_balancing.originals import truncate_sample_size, sampling_class_portion
from .utils import sample_data, compare_tuples

def test_full_balanced():
    training, testing = sample_data()
    np.random.seed(42)
    X_train, y_train, _, _ = truncate_sample_size(*training, max_size_given=load_balanced()["max"])
    original_balanced_training = (X_train, y_train)
    np.random.seed(42)
    X_test, y_test, _, _ = sampling_class_portion(*testing, class_portion=load_full_balanced())
    original_balanced_testing = (X_test, y_test)
    np.random.seed(42)
    balanced_training, balanced_testing = full_balanced(training, testing)
    assert compare_tuples(original_balanced_training, balanced_training)