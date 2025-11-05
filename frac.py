import numpy as np

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
    for n in range(2,10):
        pairings = {}
        denom = to_denom(n)
        parities = np.zeros((denom+1, n-2), dtype=bool)
        for a in range(denom+1):
            for m in range(2,n):
                m_denom = to_denom(m)
                count = (a * m_denom) // denom
                parity = count % 2 != 0
                parities[a, m-2] ^= parity

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
            print(f"{a:3}/{denom} {parity_str(parities[a])} {pairings[a]:3}    {dif}")
        print("-------")

if __name__ == "__main__":
    # print(is_reduced(2,2))
    main()

