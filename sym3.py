from sympy import symbols, factor_list, discriminant, lambdify, nroots
from matplotlib import pyplot as plt
import numpy as np

def palette(zs: np.ndarray) -> np.ndarray:
    ang = np.atan2(np.imag(zs), np.real(zs)) / 6.28318530718
    ang -= np.floor(ang)
    return np.stack([ang,ang,ang],axis=2)

def main():
    z,c = symbols("z c")
    running = z
    p = [None]
    d = [None]
    df = [[]]
    recomb = [0]
    rootss = [[]]

    fig, ax = plt.subplots(2, 3)

    for i in range(1, 7):
        running = running * running + c
        p.append(running - z)
        d.append(discriminant(running - z))
        df.append(factor_list(d[-1]))

        r = 1
        rts = []
        for factor, power in df[-1][1]:
            print(power, factor)
            r = r * factor

            rts = nroots(factor)
            print(rts)
        print()
        # print(r)
        recomb.append(r)
        rootss.append(rts)


    xs = np.linspace(-2, 2, 400, dtype=np.complex64)
    ys = np.linspace(-2, 2, 400, dtype=np.complex64)
    cs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j)

    for i in range(1, 7):
        zs = lambdify(c, recomb[i], "numpy")(cs)
        rgb = palette(zs)

        px = (i - 1) % 3
        py = (i - 1) // 3
        ax[py,px].imshow(rgb, extent=(-1,1,-1,1))

    plt.show()

if __name__ == "__main__":
    main()