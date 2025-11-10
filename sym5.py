from collections import defaultdict
from dataclasses import dataclass
from sympy import symbols, factor, factor_list, discriminant, lambdify, nroots, poly, resultant, I, Poly, symmetrize, Symbol, symmetric_poly
from matplotlib import pyplot as plt
import numpy as np

@dataclass
class Info:
    poly: Poly
    period: int

def palette(zs: np.ndarray) -> np.ndarray:
    ang = np.atan2(np.imag(zs), np.real(zs)) / 6.28318530718
    ang -= np.floor(ang)
    return np.stack([ang,ang,ang],axis=2)

def symmetrize_with_coeffs(coef: Poly, roots: list[Symbol], coefs_decreasing: list[Poly]) -> Poly:
    expr, probably0, defs = symmetrize(coef, formal=True)
    assert probably0 == 0
    result = expr
    sign = -1
    for i, (sym,definition) in enumerate(defs):
        s_i = symbols(f"s{i+1}")
        assert sym == s_i
        assert definition == symmetric_poly(i+1, *roots)
        result = result.subs(sym, sign * coefs_decreasing[i+1])
        sign *= -1
    return result

def product_of_roots_given_x_satisfies(fac: Poly, xp: Poly) -> Poly:
    """
    fac: poly in c and x
    xp: poly in x
    result: poly in c
    """
    c,x = symbols("c x")
    fac = poly(fac, c)
    xp = poly(xp, x)

    roots = []
    n = xp.degree()
    sympoly = poly(1, x)
    facprod = poly(1, c)
    for i in range(n):
        roots.append(symbols(f"r{i}"))
        sympoly *= x - roots[i] 
        facprod *= fac.subs(x, roots[i])

    xp_coefs = xp.all_coeffs()
    assert(len(xp_coefs) == n+1)
    result = 0
    for pow, coef in enumerate(reversed(facprod.all_coeffs())):
        co = symmetrize_with_coeffs(coef, roots, xp_coefs)
        result += co * (c ** pow)
    return poly(result, c)

def roots_of_unity_poly(n):
    x = symbols("x")
    result = poly(x ** n - 1)
    for m in range(1, n):
        if n % m == 0:
            p = roots_of_unity_poly(m)
            result, rem = result.div(p)
            assert rem == 0
    return result

def factor_test(disc: Poly, expected: Poly) -> tuple[Poly, int]:
    result = 0
    p = disc
    assert expected.degree() > 0
    for _ in range(100):
        div, rem = p.div(expected)
        if rem == 0:
            result += 1
            p = div
        else:
            return p, result
    raise NotImplementedError("Too many")

def main():
    MAX = 4
    z,c,x = symbols("z c x")
    running = z

    fig, ax = plt.subplots(2, 3)
    xs = np.linspace(-2, 2, 400, dtype=np.complex64)
    ys = np.linspace(-2, 2, 400, dtype=np.complex64)
    cs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j)

    expecting = defaultdict(list)
    seen_list = []
    for i in range(1, MAX+1):
        print()
        print("=================")
        print(f"i = {i}")
        print("=================")
        
        running = running * running + c

        new_seen_list = []
        disc = poly(discriminant(running - z, z), c)
        for expected, expected_degree in expecting[i]:
            disc, actual_degree = factor_test(disc, expected)
            print(f"[{expected_degree} vs {actual_degree}] of {expected}")
            new_seen_list.append(expected)

        for remaining,deg in factor_list(disc)[1]:
            print(f"Remaining discriminant [{deg}]: {remaining}")
            new_seen_list.append(remaining)
        print("---")

        if 2*i <= MAX:
            rd = poly(running, z).diff()
            res = resultant(running - z, rd - x, z)

            for fac,_ in factor_list(res)[1]:
                print(f"resultant factor {fac}")

                # check if we've seen it
                seen_any = False
                seen_all = True
                for dfac,_ in factor_list(fac.subs(x,1))[1]:
                    seen_this = dfac in seen_list
                    if seen_this:
                        seen_any = True
                    else:
                        seen_all = False
                    print(f" dfac, seen={seen_this}:  {dfac}")

                print(f" Seen any: {seen_any}. Seen all: {seen_all}")
                if seen_all:
                    continue

                for degree in range(1, MAX+1):
                    if i * degree > MAX:
                        continue
                    px = roots_of_unity_poly(degree)
                    print(f" degree = {degree}, roots_of_unity_p = {px}")
                    expected = product_of_roots_given_x_satisfies(fac, px)
                    expected_degree = i + degree if degree > 1 else i
                    for j in range(1, MAX+1):
                        if i * j * degree <= i or i * j * degree > MAX:
                            continue
                        expecting[i*j*degree].append((expected, expected_degree))
                        print(f"   --> {i*j*degree} [{expected_degree}]  {expected}")
                        faclist = factor_list(expected)[1]
                        if len(faclist) > 1:
                            print(f"   It factors!!   {faclist}")
        seen_list += new_seen_list

    # plt.show()

if __name__ == "__main__":
    main()