import numpy as np

signal = np.array([-1, 0, 0, 1, 1, 1, 0, -1, -1, 0, 0, -1])
filter_1 = np.array([1, 1, 0])
cross_correlation = np.correlate(filter_1, signal)
max_index = np.argmax(cross_correlation)
print(signal)
print(filter_1)
print(cross_correlation)
print(max_index)
