import numpy as np
from matplotlib import pyplot as plt

def to_denom(n:int) -> bool:
    return (2**n) - 1

def is_reduced(a:int,n:int) -> bool:
    for m in range(1,n):
        if n % m != 0:
            continue
        div = to_denom(n) // to_denom(m)
        if a % div == 0:
            return False
    return True

def parity_str(p: np.ndarray) -> str:
    result = ""
    for item in p:
        if item:
            result += "x"
        else:
            result += "."
    return result

def main():
    rangemax = 12
    for n in range(2,rangemax+1):
        pairings = {}
        denom = to_denom(n)
        parities = np.zeros((denom+1, n-2), dtype=bool)
        unreduced_ixs = []
        reduced_ixs = []
        for a in range(denom+1):
            if is_reduced(a,n):
                reduced_ixs.append(a)
            else:
                unreduced_ixs.append(a)
            for m in range(2,n):
                m_denom = to_denom(m)
                count = (a * m_denom) // denom
                parity = count % 2 != 0
                parities[a, m-2] ^= parity

        reduced_ixs = np.array(reduced_ixs)
        full_parity_eq = (parities.reshape(1,denom+1,n-2) == parities.reshape(denom+1,1,n-2)).all(axis=-1)
        parity_eq = full_parity_eq[:,reduced_ixs]
        if n <= 9:
            for row in parity_eq[reduced_ixs[:len(reduced_ixs)//2],:]:
                print(parity_str(row[:len(row)//2]))
        for a in range(1,denom):
            if not is_reduced(a,n):
                print(f"{a:3}/{denom}                  not_reduced")
                continue
            if a not in pairings:
                for b in range(a+1, denom):
                    if not is_reduced(b,n):
                        continue
                    if np.array_equal(parities[b], parities[a]):
                        pairings[a] = b
                        pairings[b] = a
                        break
                assert a in pairings
            if abs(pairings[a] - a) > 1:
                dif = f"{pairings[a]-a:+3}"
            else:
                dif = "   "
            n_eq = parity_eq[a,:].sum()

            print(f"{a:3}/{denom} {parity_str(parities[a])} {pairings[a]:3}    {dif}    {n_eq:3}")
        print("-------")

        if n == rangemax:
            im = np.array(full_parity_eq, dtype=np.int8) + 1
            for a,b in pairings.items():
                assert im[a,b] == 2
                im[a,b] = 5
            for u in unreduced_ixs:
                im[u,:] = 0
                im[:,u] = 0
            plt.imshow(im)
            plt.show()

if __name__ == "__main__":
    # print(is_reduced(2,2))
    main()

