from random import randint

def generate_copy_task(length, samples, max_val):
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
        y_i = X_i.copy()
        X.append(X_i)
        y.append(y_i)
    return X, y

def generate_single_task(length, samples, max_val, return_index=0):
    assert return_index < length
    X = []
    y = []
    for _ in range(samples):
        X_i = [[randint(0, max_val)] for _ in range(length)]
        y_i = [[X_i[return_index]]]
        X.append(X_i)
        y.append(y_i)
    return X, y
