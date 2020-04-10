import numpy as np
import pandas as pd
from source.tools import pca_tools as pt
from source.tools import random_forest_tools as rt
from source.tools import model_aid_tools as mt
from collections import Counter

a = np.array([[1, 2, 321, 31],
              [3, 4, 43, 12],
              [3, 11, 23, 56]
              ])

b = np.sum(np.power(a, 2), axis=1)
c = np.sum(np.power(a, 2), axis=1).reshape(-1, 1)
x = a[1:, :]

df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'])

# rt.information_of_feature(a, label=None, index=1)
d = {'a':1,'b':[2]}
print(len(d.get('c', [])))
0
