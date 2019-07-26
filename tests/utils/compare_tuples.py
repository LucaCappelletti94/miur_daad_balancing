import numpy as np
from typing import Tuple

def compare_tuples(A:Tuple[np.ndarray, np.ndarray], B:Tuple[np.ndarray, np.ndarray])->bool:
    """Return a boolean checking if the two given arrays are equal."""
    return all([
        np.all(a==b) for a, b in zip(A, B)
    ])