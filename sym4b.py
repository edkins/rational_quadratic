from sympy import symbols, factor, factor_list, discriminant, lambdify, nroots, poly, resultant, I
from matplotlib import pyplot as plt
import numpy as np

def palette(zs: np.ndarray) -> np.ndarray:
    ang = np.atan2(np.imag(zs), np.real(zs)) / 6.28318530718
    ang -= np.floor(ang)
    return np.stack([ang,ang,ang],axis=2)

def main():
    z,c,x = symbols("z c x")
    running = z

    fig, ax = plt.subplots(2, 3)
    xs = np.linspace(-2, 2, 400, dtype=np.complex64)
    ys = np.linspace(-2, 2, 400, dtype=np.complex64)
    cs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j)

    for i in range(1, 6):
        print("=================")
        print(f"i = {i}")
        print("=================")
        
        running = running * running + c
        rd = poly(running, z).diff()

        px = (i - 1) % 3
        py = (i - 1) // 3

        res = resultant(running - z, rd - x, z)

        print("x = x")
        for fac,_ in factor_list(res)[1]:
            print(fac)
        print()
        print("x = 1")
        for fac,_ in factor_list(res.subs(x,1))[1]:
            print(fac)
        print()
        print("x = -1")
        for fac,_ in factor_list(res.subs(x,-1))[1]:
            print(fac)
        print()
        print("x = I")
        for fac,_ in factor_list(res.subs(x,I))[1]:
            print(fac)
        print()
        print()

    # plt.show()

if __name__ == "__main__":
    main()