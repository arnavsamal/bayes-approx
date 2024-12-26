from sklearn.datasets import make_classification
import numpy as np

def generate_dataset():
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    y = 2 * y - 1  # Convert labels to -1 and 1 for binary classification
    return X, y