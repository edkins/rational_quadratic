from sympy import symbols, factor, factor_list, discriminant, lambdify, nroots, poly, resultant
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
        d = poly(discriminant(running - z), c)
        rd = poly(running, z).diff()

        px = (i - 1) % 3
        py = (i - 1) // 3

        for dfac,_ in factor_list(d)[1]:
            rts = np.roots(dfac.all_coeffs())
            # rts = np.array(nroots(d, n=4, maxsteps=1000), dtype=np.complex64)
            ax[py,px].scatter(np.real(rts), np.imag(rts), s=20)
        # print(rts)

        res = resultant(running - z, rd - x, z)
        # print(poly(res,z).all_coeffs())

        for fac,_ in factor_list(res)[1]:
            # print(res)
            res_coef_lambdas = [
                lambdify(x,p) for p in reversed(poly(fac, c).all_coeffs())
            ]
            # print(res_coef_lambdas)

            n_xs = 17
            n_roots = len(res_coef_lambdas) - 1
            points = np.zeros(n_xs * n_roots, dtype=np.complex64)
            for xi,theta in enumerate(np.linspace(0, 6.28318530718, n_xs)):
                xnum = np.cos(theta) + np.sin(theta) * 1j
                res_coef_num = np.array([f(xnum) for f in res_coef_lambdas], dtype=np.complex64)
                # these_roots = np.roots(res_coef_num)

                # poly_num = sum(coef * (c ** n) for n,coef in enumerate(res_coef_num))
                # these_roots = nroots(poly_num, n=3, maxsteps=1000)
                
                poly_num = np.polynomial.Polynomial(res_coef_num)
                these_roots = poly_num.roots()
                # poly_num.domain = np.array([-2,2])
                # print(poly_num)

                assert len(these_roots) == n_roots
                points[xi*n_roots:(xi+1)*n_roots] = these_roots

            ax[py,px].scatter(np.real(points), np.imag(points), s=1)
        ax[py,px].set_xlim(-2,2)
        ax[py,px].set_ylim(-2,2)
        ax[py,px].axis("equal")
        print()

    plt.show()

if __name__ == "__main__":
    main()