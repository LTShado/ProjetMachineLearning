import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import *

PATH_TO_SHARED_LIBRARY = "MachineLearningLib/x64/Debug/MachineLearningLib.dll"

points = np.array([
    [1, 1],
    [2, 1],
    [2, 2]
])
classes = np.array([
    1,
    1,
    -1
])

colors = ['blue' if c == 1 else 'red' for c in classes]

W = np.random.uniform(-1.0, 1.0, 3)
test_points = []
test_colors = []
for row in range(0, 300):
    for col in range(0, 300):
        p = np.array([col / 100, row / 100])
        c = 'lightcyan' if np.matmul(np.transpose(
            W), np.array([1.0, *p])) >= 0 else 'pink'
        test_points.append(p)
        test_colors.append(c)
test_points = np.array(test_points)
test_colors = np.array(test_colors)

plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
plt.scatter(points[:, 0], points[:, 1], c=colors)
plt.show()

if __name__ == "__main__":
    # Load lib
    my_lib = cdll.LoadLibrary(PATH_TO_SHARED_LIBRARY)
    my_lib.create_array.argtypes = [c_int]
    my_lib.create_array.restype = POINTER(c_double)
