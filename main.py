import sys
import pandas as pd
import numpy as np
from svd import svd


if __name__ == "__main__":
    ratings_path = sys.argv[1]
    targets_path = sys.argv[2]
    svd_ = svd(epochs = 20, lr = 0.005, n_factors=5, reg=0.1)
    svd_.read_ratings(ratings_path)
    svd_.stochastic_gradient()
    svd_.submission(targets_path)