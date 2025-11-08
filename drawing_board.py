from matplotlib import pyplot as plt
import numpy as np

MAXITER=50
BAILOUT=1024
WIDTH=400
HEIGHT=300

def main():
    xmin = -1.5
    xmax = 0.5
    ymin = 0
    ymax = 1.5
    xs = np.linspace(xmin, xmax, WIDTH, dtype=np.complex128)
    ys = np.linspace(ymin, ymax, HEIGHT, dtype=np.complex128)
    cs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j).reshape(-1)
    npixels = cs.shape[0]

    cabs = np.abs(cs)
    cs[cabs > 2] = cs[cabs > 2] * ((cabs[cabs>2]/2) ** 4)

    stuff = np.zeros((cs.shape[0], MAXITER), dtype=np.complex128)

    zs = np.array(cs)
    still_going = np.ones_like(zs, dtype=bool)
    iters = np.zeros_like(zs, dtype=np.int32)
    for i in range(MAXITER):
        stuff[still_going,i] = zs
        iters[still_going] = i
        zs *= zs
        zs += cs

        ok = abs(zs) < BAILOUT
        zs = zs[ok]
        cs = cs[ok]
        still_going[still_going] = ok

    iters[still_going] = MAXITER
    stuff[iters == MAXITER,:] = 0
    # for i in range(MAXITER):
    #     stuff[iters == i,:i+1] = stuff[iters == i,i::-1]

    fig, ax = plt.subplots(4, 6, squeeze=False, figsize=(25,15))
    pic = np.zeros_like(stuff[:,0], dtype=np.float64)
    for i in range(12):
        py = 2*(i//6)
        px = i%6
        angles = np.atan2(np.imag(-stuff[:,i]), np.real(-stuff[:,i])) / 6.28318530718
        ax[py,px].imshow(angles.reshape(HEIGHT,WIDTH), extent=(xmin,xmax,ymin,ymax), vmin=-0.5, vmax=0.5)
        ax[py,px].axis("off")

        sel = iters >= i
        pic *= 2
        pic[sel] = np.floor(pic[sel] - angles[sel]) + angles[sel] + 0.5
        ax[py+1,px].imshow(pic.reshape(HEIGHT,WIDTH), extent=(xmin,xmax,ymin,ymax))
        ax[py+1,px].axis("off")
    
    # ax[3,3].imshow(final_pic.reshape(HEIGHT,WIDTH), extent=(xmin,xmax,ymin,ymax))
    # ax[3,3].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
