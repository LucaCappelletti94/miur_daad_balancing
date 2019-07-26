from typing import Tuple
import numpy as np

def balanced(training:np.ndarray, testing:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """Return balanced training data to the given maximum, leaving testing untouched."""
    X_train, y_train = training
    unique, counts = np.unique(y_train, return_counts=True)
    