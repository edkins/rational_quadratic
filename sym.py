from sympy import symbols, simplify, Eq, solve, poly, factor, cancel, rootof, expand

z, a, b, j, k= symbols('z a b j k')

def f(x):
    return (a * x * x + b * x) / ((a + b - 1) * x + 1)

assert f(0) == 0
assert f(1) == 1

def fdash(x):
    return f(z).diff(z).subs(z, x)
# print(fdash(z))
assert fdash(0) == b
eq1 = Eq(j, (a + b - 1) / a)    # fdash(infinity)
eq2 = Eq(k, simplify(fdash(1)))

a_bj = simplify(solve(eq1, a)[0])
eq2 = simplify(eq2.subs(a, a_bj))

print(eq2)
k_bj = simplify(solve(eq2, k)[0])
print(k_bj)

assert simplify(b * j * k_bj) == simplify(b + j + k_bj - 2)

print("=========================")

eq0 = factor(cancel(f(f(z)) - z))
print(eq0)

num, den = simplify(eq0).as_numer_denom()
p = poly(num / (z * (z - 1) * (b - 1)), z)

print(p)
p2 = poly(simplify(p.subs(a, a_bj) * (j-1)*(j-1)), z)

print(p2)
r0 = rootof(p, z, 0)
r1 = rootof(p, z, 1)
print(simplify(r0 + r1))
print(simplify(r0 * r1))
print(simplify((r0 + r1) / (r0 * r1)))
assert simplify(r0 + r1) == (-b-1)/a
assert simplify(a * (r0 + r1) - (r0 + r1) / (r0 * r1) - 2 * a + 2) == 0

s0, s1 = symbols('s0 s1')
asolv = solve(a * (s0 + s1) - (s0 + s1) / (s0 * s1) - 2 * a + 2, a)[0]
print(asolv)
bsolv = solve(s0 + s1 + (b+1) / asolv, b)[0]
print(bsolv)

ffs = f(f(z)).subs({a: asolv, b: bsolv})
print(factor(ffs - z))

# print(eq0.subs(z, 0))
# print(eq0.subs(z, 1))
# # eq0b = eq0.normal() / (z * z - z)
# eq0n = simplify(simplify(eq0).as_numer_denom()[0])
# # eq0nb = simplify(eq0n / (z * (z - 1)))
# print(eq0n.subs(z, 0))
# print(eq0n.subs(z, 1))
# print(poly(eq0n, z))