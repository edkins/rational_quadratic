from typing import Any
from matplotlib import pyplot as plt
import numpy as np

JULIA=True

if JULIA:
    MAXITER=26
    MANYPLOTS=False
    FIRST_PLOT=22
else:
    MAXITER=200
    MANYPLOTS=False
    FIRST_PLOT=0

ALMOST_BAILOUT=900
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
        xmin = -5
        xmax = 5
        ymin = -5
        ymax = 5
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
        # c = -1.025 + 0.260j
        c = -.2 + .826j
        xs = np.linspace(xmin, xmax, width, dtype=np.complex128)
        ys = np.linspace(ymax, ymin, height, dtype=np.complex128)
        zs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j).reshape(-1)
        cs = np.full_like(zs, c, dtype=np.complex128)
    else:
        c = 0
        xs = np.linspace(xmin, xmax, width, dtype=np.complex128)
        ys = np.linspace(ymax, ymin, height, dtype=np.complex128)
        cs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j).reshape(-1)
        zs = np.array(cs)

    z0s = np.array(zs)

    rgb = np.zeros((cs.shape[0], 3), dtype=np.float32)
    still_going = np.ones_like(zs, dtype=bool)
    iters = np.zeros_like(zs, dtype=np.int32)
    for i in range(MAXITER):
        iters[still_going] = i
        zs *= zs
        zs += cs

        view = np.array(still_going)
        view[still_going] = (abs(zs) >= BAILOUT) & (np.imag(zs) > 0)
        rgb[view] = [0.0,0.0,0.5]

        ok = abs(zs) < BAILOUT
        zs = zs[ok]
        cs = cs[ok]
        still_going[still_going] = ok

        # view = np.array(still_going)
        # view[still_going] = abs(zs) >= ALMOST_BAILOUT
        # rgb[still_going][abs(zs) >= ALMOST_BAILOUT] = [1.0,1.0,1.0]    # This kind of double indexing
        # rgb[view] = [1.0,1.0,0.0]

    im = np.imag(z0s / np.sqrt(c))
    re = np.real(z0s / np.sqrt(c))
    # angles = frac(np.atan2(im * im, re) / 6.28318530718)
    angles = frac((np.atan2(np.sqrt(1 + 1/re/re) * im, re) + np.atan2(c.imag, c.real)/2) / 6.28318530718)
    rgb[(angles > 0.99) | (angles < 0.01),1] = 1
    for i in range(8):
        rgb[(angles > i/8-0.01) & (angles < i/8+0.01),1] = 1

    iters[still_going] = MAXITER
    ax[0,0].imshow(rgb.reshape(height,width,3), extent=(xmin,xmax,ymin,ymax))
    ax[0,0].axis("off")
    ax[0,0].set_ylim(full_ymin, full_ymax)

    if JULIA:
        ys0 = np.linspace(ymin, ymax, 100)
        xs0 = np.sqrt(1 + ys0 * ys0)
        
        # angle = np.atan2(np.imag(c), np.real(c)) / 2
        # angle = 0
        angle = 6.28318530718 / 8

        ys = xs0 * np.sin(angle) + ys0 * np.cos(angle)
        xs = -xs0 * np.cos(angle) + ys0 * np.sin(angle)

        ax[0,0].plot(xs, ys)

if __name__ == "__main__":
    main()
