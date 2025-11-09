from typing import Any
from matplotlib import pyplot as plt
import numpy as np

JULIA=False

if JULIA:
    MAXITER=32
    MANYPLOTS=True
    FIRST_PLOT=22
    INVESTIGATE=False
else:
    MAXITER=200
    MANYPLOTS=False
    FIRST_PLOT=0
    INVESTIGATE=False

BAILOUT=1024
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
        mapping = {
            (FIRST_PLOT,0): (0,0),
            (FIRST_PLOT,1): (1,0),
            (FIRST_PLOT+1,0): (0,1),
            (FIRST_PLOT+1,1): (1,1),
            (FIRST_PLOT+2,0): (0,2),
            (FIRST_PLOT+2,1): (1,2),
            (MAXITER-1,2): (0,3),
            (MAXITER-1,1): (1,3),
        }
    elif INVESTIGATE:
        mapping = {
            (22,0): (0,0),
            (22,1): (1,0),
            (23,0): (0,1),
            (23,1): (1,1),
        }
    else:
        mapping = {
            (MAXITER-1,2): (0,0)
        }

    fig, ax = plt.subplots(1+max(y for y,x in mapping.values()), 1+max(x for y,x in mapping.values()), squeeze=False, figsize=(25,15))

    num_tiles = 1 + MAXITER // 50
    for tile in range(num_tiles):
        do_plot(tile, num_tiles, ax, mapping)
    plt.tight_layout()
    plt.show()

def do_plot(tile: int, num_tiles: int, ax: Any, mapping: dict):
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
        width = 2000 if MANYPLOTS else 2000
        height = 2000 if MANYPLOTS else 2000
        if INVESTIGATE:
            xmin = -0.35
            xmax = -0.15
            ymin = 0.1
            ymax = 0.3
        else:
            xmin = -1.7
            xmax = 1.7
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
        c = -1.025 + 0.260j
        # c = -.2 + .826j
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
        im = np.imag(stuff[:,i])
        re = np.real(stuff[:,i])
        length = np.abs(stuff[:,i])
        angles = frac(np.atan2((1 - 0.5 / (1 + length * length)) * im, re) / 6.28318530718)
        if (i,0) in mapping:
            py,px = mapping[(i,0)]
            # pic1 = frac(angles * (1 + np.float64((iters % 2 == 0))))
            ax[py,px].imshow(angles.reshape(height,width), extent=(xmin,xmax,ymin,ymax), vmin=0, vmax=1)
            ax[py,px].axis("off")
            ax[py,px].set_ylim(full_ymin, full_ymax)

        sel = iters >= i
        if i == 0:
            pic = np.array(angles)
        else:
            pic[sel] = frac(0.5 * (pic + np.floor(2.0 * angles - pic + 0.5))[sel])

        if (i,1) in mapping:
            py,px = mapping[(i,1)]
            # pic2 = frac(pic * (1 + np.float64((iters % 2 == 0) & sel)))
            ax[py,px].imshow(pic.reshape(height,width), extent=(xmin,xmax,ymin,ymax), vmin=0, vmax=1)
            ax[py,px].axis("off")
            ax[py,px].set_ylim(full_ymin, full_ymax)

        if (i,2) in mapping:
            rgb = pic.reshape(-1,1).repeat(3, axis=1)
            for num,denom,j in fractions:
                rgb[(pic >= num/denom-0.0001) & (pic <= num/denom+0.0001),:] = PALETTE[j]

            py,px = mapping[(i,2)]
            ax[py,px].imshow(rgb.reshape(height,width,3), extent=(xmin,xmax,ymin,ymax))
            ax[py,px].axis("off")
            ax[py,px].set_ylim(full_ymin, full_ymax)

if __name__ == "__main__":
    main()
