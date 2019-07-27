from miur_daad_balancing import get_classes
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split

def odd_even_split(data:List)->Tuple[List, List]:
    """Return given list split into even and odds elements.
        data:List, list of data to split.
    """
    return data[::2], data[1::2]

def sample_data()->Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return Tuple with training and testing data and classes"""
    X1 = np.random.randint(0, 10, size=(50000, 26))
    X2 = np.random.randint(0, 10, size=(50000, 200, 5))
    y = np.random.choice(get_classes(), size=X1.shape[0])
    return odd_even_split(train_test_split(X1, X2, y, test_size=0.3, random_state=42))