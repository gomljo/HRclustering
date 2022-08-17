from sklearn.datasets import make_classification, make_blobs, make_circles, make_moons

import pandas as pd
import numpy as np
X, y = make_moons(500, random_state=42)
y = y.reshape(-1, 1)
print(X.shape)
print(y.shape)

df1 = pd.DataFrame(X, columns=['x1', 'x2'])
print(df1)
df2 = pd.DataFrame(y, columns=['target'])
print(df2)
df = pd.concat([df1, df2], axis=1)
print(df)

a = np.arange(0, 100)

