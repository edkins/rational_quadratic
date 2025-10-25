from matplotlib import pyplot as plt
import numpy as np

lower = -5
upper = 5
a = np.linspace(lower, upper, 400).reshape((-1, 1))
b = np.linspace(lower, upper, 400).reshape((1, -1))
a=1-1/a
b=1-1/b
c = (b + a - 2) / (b * a - 1)
c=1/(1-c)

fig, ax = plt.subplots(squeeze=False)

ax[0,0].imshow(c, extent=(lower, upper, lower, upper), origin='lower', vmin=lower, vmax=upper)
plt.show()
