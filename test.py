import numpy as np
from source.tools import pca_tools as pt

a = np.array([[1, 2, 321, 31],
              [3, 4, 43, 12],
              [3, 11, 23, 56]
              ])

b = np.sum(np.power(a, 2), axis=1)
c = np.sum(np.power(a, 2), axis=1).reshape(-1, 1)
x = a[1:, :]

