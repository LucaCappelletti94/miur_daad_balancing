from miur_daad_balancing import get_classes
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

def sample_data()->Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return Tuple with training and testing data and classes"""
    X = np.random.randint(0, 10, size=(50000, 26))
    y = np.random.choice(get_classes(), size=X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (X_train, y_train), (X_test, y_test)