from sympy import symbols, simplify, poly, rootof

z, a, b, c = symbols('z a b c')

b_ = 1 - 1 / b
c_ = 1 - 1 / c

B = b_
A = simplify((1 - b_ * c_) / (c_ - 1))
print("A=", A)
print("A+B-1=", simplify(A + B - 1))


def f(x):
    # return ((1-b-c) * x * x + (b-1) * x) / ((-b-c) * x + b)
    return (a * x * x + (b-1) * x) / ((a-1) * x + b)

assert simplify(f(0)) == 0
assert simplify(f(1)) == 1

fdashz = f(z).diff(z)
print("f'(0) = ", simplify(fdashz.subs(z, 0)))
print("f'(1) = ", simplify(fdashz.subs(z, 1)))

p = poly(simplify(fdashz.normal()).as_numer_denom()[0], z)
print(p)
r0 = rootof(p, z, 0)
r1 = rootof(p, z, 1)
print(simplify(fdashz.subs(z, r0)))
print(simplify(fdashz.subs(z, r1)))