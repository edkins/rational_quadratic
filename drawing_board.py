from matplotlib import pyplot as plt
import numpy as np

MAXITER=200
BAILOUT=1024
JULIA=False
MANYPLOTS=False

PALETTE = [
    None,
    [1,0,0],
    [0,1,0],
    [1,1,0],
    [0,0,1],
    [1,0,1],
    [0,1,1],
    [1,0.5,0],
    [0.7,0.5,1],
    [1,0,0],
]

def frac(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)

def main():
    if MANYPLOTS:
        fig, ax = plt.subplots(4, 6, squeeze=False, figsize=(25,15))
    else:
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(25,15))

    num_tiles = 1 + MAXITER // 50
    for tile in range(num_tiles):
        do_plot(tile, num_tiles, ax)
    plt.tight_layout()
    plt.show()

def do_plot(tile: int, num_tiles: int, ax: np.ndarray):
    fractions = []
    for j in range(1,8):
        denom = (1<<j)-1
        for num in range(denom+1):
            found = False
            for n2,d2,_ in fractions:
                if n2*denom == num*d2:
                    found = True
                    break
            if not found:
                fractions.append((num,denom,j))
    fractions.sort(key=lambda x:x[0]/x[1])

    if JULIA:
        width = 400 if MANYPLOTS else 2000
        height = 400 if MANYPLOTS else 2000
        xmin = -1.5
        xmax = 1.5
        ymin = -1.5
        ymax = 1.5
    else:
        width = 700 if MANYPLOTS else 2800
        height = 300 if MANYPLOTS else 1200
        xmin = -2.25
        xmax = 1.25
        ymin = 0
        ymax = 1.5

    full_ymin = ymin
    full_ymax = ymax
    if num_tiles > 1:
        start = (height * tile) // num_tiles
        end = (height * (tile + 1)) // num_tiles

        ymin = ymin + start * (ymax-ymin) / height
        ymax = ymin + end * (ymax-ymin) / height
        height = end - start

    print(f"Drawing tile {tile}/{num_tiles}. ymin={ymin}, ymax={ymax}, height={height}")

    if JULIA:
        # c = -0.06685714285714286 + 0.98j
        c = -1.025 + 0.260j
        xs = np.linspace(xmin, xmax, width, dtype=np.complex128)
        ys = np.linspace(ymax, ymin, height, dtype=np.complex128)
        zs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j).reshape(-1)
        cs = np.full_like(zs, c, dtype=np.complex128)
    else:
        xs = np.linspace(xmin, xmax, width, dtype=np.complex128)
        ys = np.linspace(ymax, ymin, height, dtype=np.complex128)
        cs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j).reshape(-1)
        zs = np.array(cs)

    stuff = np.zeros((cs.shape[0], MAXITER), dtype=np.complex128)

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
    for i in range(MAXITER):
        stuff[iters == i,:i+1] = stuff[iters == i,i::-1]

    pic = np.zeros_like(stuff[:,0], dtype=np.float64)
    for i in range(MAXITER):
        py = 2*(i//6)
        px = i%6
        angles = frac(np.atan2(np.imag(stuff[:,i]), np.real(stuff[:,i])) / 6.28318530718)
        if MANYPLOTS and i < 12:
            ax[py,px].imshow(angles.reshape(height,width), extent=(xmin,xmax,ymin,ymax), vmin=0, vmax=1)
            ax[py,px].axis("off")
            ax[py,px].set_ylim(full_ymin, full_ymax)

        sel = iters >= i
        if i == 0:
            pic = np.array(angles)
        else:
            pic[sel] = frac(0.5 * (pic + np.floor(2.0 * angles - pic + 0.5))[sel])

        if MANYPLOTS and i < 12:
            ax[py+1,px].imshow(pic.reshape(height,width), extent=(xmin,xmax,ymin,ymax), vmin=0, vmax=1)
            ax[py+1,px].axis("off")
            ax[py+1,px].set_ylim(full_ymin, full_ymax)
    
    rgb = pic.reshape(-1,1).repeat(3, axis=1)
    for num,denom,j in fractions:
        rgb[(pic >= num/denom-0.0001) & (pic <= num/denom+0.0001),:] = PALETTE[j]
    ax[-1,-1].imshow(rgb.reshape(height,width,3), extent=(xmin,xmax,ymin,ymax))
    ax[-1,-1].axis("off")
    ax[-1,-1].set_ylim(full_ymin, full_ymax)

if __name__ == "__main__":
    main()
