import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def gen_filter(start, end, n):
    interp = interpolate.interp1d([0, 0.5, 1], [0, 1, 0])
    fraction = end-start
    new_scale = np.linspace(0, 1, n*fraction)
    return np.concatenate((np.zeros(int((start-0)*n)), interp(new_scale), np.zeros(int((1-end)*n))))

f = np.arange(0, 4000, 1)
filter = gen_filter(0.1, 0.2, 2000)

fig, axes = plt.subplots(2, 1)
axes[0].plot([0, 0.5, 1], [0, 1, 0])
axes[1].plot(filter, '-')
plt.show()