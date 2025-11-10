from dataclasses import dataclass
from typing import Literal, Optional
from sympy import symbols, factor, factor_list, discriminant, lambdify, nroots, poly, resultant, I, Poly, symmetrize, Symbol, symmetric_poly
import numpy as np

@dataclass
class Info:
    poly: Poly
    respoly: Optional[Poly]
    period: int
    seen_period: int
    degree: int
    status: Literal["seen_in_discriminant","seen_in_resultant","expanded"]
    parent: Optional[Poly]

    def expected_degree(self, period: int) -> int:
        # TODO
        if self.period_divides(period):
            return self.degree
        else:
            return 0
    
    def period_divides(self, period: int) -> bool:
        return self.period <= period and period % self.period == 0

    def dpoly_for_period(self, period: int) -> Poly:
        assert self.period_divides(period)
        if period == self.period:
            return self.poly
        assert self.respoly is not None
        rou = roots_of_unity_poly(period // self.period, full=True)
        result = product_of_roots_given_x_satisfies(self.respoly, rou)
        # print(f"{self}.dpoly_for_period[{period}] == {result}")
        return result

def palette(zs: np.ndarray) -> np.ndarray:
    ang = np.atan2(np.imag(zs), np.real(zs)) / 6.28318530718
    ang -= np.floor(ang)
    return np.stack([ang,ang,ang],axis=2)

def symmetrize_with_coeffs(main_poly: Poly, roots: list[Symbol], coefs_decreasing: list[Poly]) -> Poly:
    expr, probably0, defs = symmetrize(main_poly, formal=True)
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

def roots_of_unity_poly(n, full:bool):
    x = symbols("x")
    result = poly(x ** n - 1)
    if full:
        return result
    for m in range(1, n):
        if n % m == 0:
            p = roots_of_unity_poly(m, full=False)
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

def find_polynomials_with_product(infos: list[Info], prodp: Poly) -> list[Info]:
    result = []
    for info in infos:
        div, rem = prodp.div(info.poly)
        if rem == 0:
            result.append(info)
            prodp = div
    assert prodp == 1
    return result

def main():
    MAX = 12
    DMAX = 6
    RMAX = 5
    z,c,x = symbols("z c x")
    running = z
    infos:list[Info] = []

    xs = np.linspace(-2, 2, 400, dtype=np.complex64)
    ys = np.linspace(-2, 2, 400, dtype=np.complex64)
    cs = (xs.reshape(1,-1) + ys.reshape(-1,1) * 1j)

    for i in range(1, DMAX+1):
        print()
        print("=================")
        print(f"period = {i}")
        print("=================")

        running = running * running + c

        print("Calculating discriminant...")
        disc = poly(discriminant(running - z, z), c)
        print("Factoring discriminant...")
        for info in infos:
            disc, actual_degree = factor_test(disc, info.poly)
            expected_degree = info.expected_degree(i)
            assert expected_degree == actual_degree
            if expected_degree != 0 or actual_degree != 0:
                print(f"[{actual_degree}] of {info.poly.as_expr()}")

        for remaining,deg in factor_list(disc)[1]:
            print(f"Remaining discriminant [{deg}]: {remaining.as_expr()}")
            infos.append(Info(poly=poly(remaining,c), respoly=None, period=i, seen_period=i, degree=i, parent=None, status="seen_in_discriminant"))

        print("---")

        if i > RMAX:
            print("Skipping resultant calculation here.")
            continue
        rd = poly(running, z).diff()
        print("Computing resultant...")
        res = poly(resultant(running - z, rd - x, z), c)
        print("Factoring resultant...")
        for fac,facdegree in factor_list(res)[1]:
            print(f"resultant factor [{facdegree}] {fac.as_expr()}")

            # corresponding discriminant poly
            corresponding_dpoly = poly(fac.subs(x,1), c)
            found = [info for info in infos if info.period_divides(i) and info.dpoly_for_period(i) == corresponding_dpoly]
            assert len(found) <= 1
            if len(found) == 0:
                found2 = find_polynomials_with_product(infos, corresponding_dpoly)
                assert all(info.period == i for info in found2)
                print(f" Found dpoly in {len(found2)} pieces: {[f.poly.as_expr() for f in found2]}")
            elif found[0].period == i:
                assert found[0].respoly is None
                found[0].respoly = fac
                print(f" Setting respoly")
            elif found[0].period < i:
                print(f" Definitely seen dpoly! (period={found[0].period})")
                continue

            if 2*i <= MAX:
                # # check if we've seen it
                # seen_any = False
                # seen_all = True
                # for dfac,_ in factor_list(fac.subs(x,1))[1]:
                #     dfacp = poly(dfac,c)
                #     seen_this = [info for info in infos if info.poly == dfacp]
                #     assert len(seen_this) == 1  # should have at least seen it in our own discriminant

                #     if seen_this[0].status == "expanded":
                #         seen_any = True
                #         print(f" dfac, seen={seen_this}")
                #     elif seen_this[0].status == "seen_in_discriminant":
                #         assert seen_this[0].period == i
                #         seen_all = False
                #         print(f" dfac, unseen[d]={dfacp}")
                #     elif seen_this[0].status == "seen_in_resultant":
                #         assert seen_this[0].seen_period < i
                #         print(f" dfac, unseen[r]={dfacp}")
                #         seen_all = False
                #     else:
                #         assert False

                # print(f" Expanded any: {seen_any}. Expanded all: {seen_all}")
                # if seen_all:
                #     continue

                for degree in range(2, MAX+1):
                    if i * degree > MAX:
                        continue
                    px = roots_of_unity_poly(degree, full=False)
                    print(f" degree = {degree}, roots_of_unity_p = {px}")
                    expected = product_of_roots_given_x_satisfies(fac, px)
                    assert not any(info.poly == expected for info in infos)
                    infos.append(Info(poly=expected, respoly=None, period=i*degree, seen_period=i, degree=i*(1+degree), parent=fac, status="seen_in_resultant"))
                    print(f"   --> {i*degree}  {expected}")
                        # expecting[i*j*degree].append((expected, expected_degree))
                        # faclist = factor_list(expected)[1]
                        # if len(faclist) > 1:
                        #     print(f"   It factors!!   {faclist}")

            for info in infos:
                if info.period == i:
                    assert info.status in ["seen_in_discriminant", "seen_in_resultant"]
                    info.status = "expanded"

if __name__ == "__main__":
    main()