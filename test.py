import numpy as np
import pandas as pd
from source.tools import pca_tools as pt

a = np.array([[1, 2, 321, 31],
              [3, 4, 43, 12],
              [3, 11, 23, 56]
              ])

b = np.sum(np.power(a, 2), axis=1)
c = np.sum(np.power(a, 2), axis=1).reshape(-1, 1)
x = a[1:, :]

df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'])

node_classification = np.array([1,1,1,1,0,0,0,0])

for i in range(2):
    index = np.where(node_classification == i)
    0

0
