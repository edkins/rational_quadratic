from matplotlib import pyplot as plt
import numpy as np

lower = -3
upper = 5
a = np.linspace(lower, upper, 400).reshape((-1, 1))
b = np.linspace(lower, upper, 400).reshape((1, -1))
a1=1-1/a
b1=1-1/b
c1 = (b1 + a1 - 2) / (b1 * a1 - 1)
c=1/(1-c1)

fig, ax = plt.subplots(squeeze=False, subplot_kw={"projection": "3d"})

ax[0,0].plot_wireframe(a, b, c, rstride=5, cstride=5, axlim_clip=True)
ax[0,0].set(xlim=(lower, upper), ylim=(lower, upper), zlim=(lower, upper))
plt.show()
