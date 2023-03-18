import numpy as np

x = np.mat([-1, -3, 3, 6, 0])
x[x < 0]=0
print(x)
