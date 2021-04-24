import numpy as np
from modAL.utils.data import retrieve_rows

def get_row(feature_matrix, idx: int):
    """Utility function that returns row @idx from @feature_matrix as a 2d np array regardless of matrix format."""
    if isinstance(feature_matrix, np.ndarray):
        return retrieve_rows(feature_matrix, [idx])
    else:
        return retrieve_rows(feature_matrix, idx)


def eq_split(X, y, n_per_class, random_state=None):
    if random_state:
        np.random.seed(random_state)
    sampled = X.groupby(y, sort=False).apply(
        lambda frame: frame.sample(n_per_class))
    mask = sampled.index.get_level_values(1)

    X_train = X.drop(mask)
    X_test = X.loc[mask]

    return X_train, X_test