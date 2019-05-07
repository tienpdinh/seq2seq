import numpy as np
from numpy.random import randint

def generate_copy_task(length, samples, max_val, reverse=False):
    """
    Generate copies of encoder input and decoder target that are identical
    :param length: The length of each sequence
    :param samples: The total number of sequences
    :param max_val: The maximum value in each sequence
    """
    X = []
    y = []
    for _ in range(samples):
        X_i = [[randint(0, max_val)] for _ in range(length)]
        y_i = X_i.copy() if not reverse else X_i.copy()[::-1]
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)

def generate_single_task(length, samples, max_val, return_index=0):
    assert return_index < length
    X = []
    y = []
    for _ in range(samples):
        X_i = [[randint(0, max_val)] for _ in range(length)]
        y_i = [X_i[return_index]]
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)

def one_hot_encode(tensor, max_val):
    # Shape should be (batch * sample * elements)
    assert len(tensor.shape) == 3
    tensor_shape = tensor.shape
    encoded_tensor = np.zeros((tensor_shape[0], tensor_shape[1], max_val))
    for i in range(tensor_shape[0]):
        for j in range(tensor_shape[1]):
            encoded_tensor[i, j, tensor[i, j, 0]] = 1.
    return encoded_tensor
