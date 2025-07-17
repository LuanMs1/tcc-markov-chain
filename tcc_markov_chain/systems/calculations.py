import numpy as np
def aperiodic_relative_distances(positions:np.ndarray, k:int=None) -> np.ndarray:
    return positions - positions[k]