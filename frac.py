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
    for n in range(2,6):
        denom = to_denom(n)
        prev_parities = np.zeros(n-2, dtype=bool)
        for a in range(denom+1):
            if not is_reduced(a,n):
                continue

            parities = np.zeros(n-2, dtype=bool)
            for m in range(2,n):
                m_denom = to_denom(m)
                count = (a * m_denom) // denom
                parity = count % 2 != 0
                parities[m-2] ^= parity
            if a > 1:
                print("    ", parity_str(parities ^ prev_parities))
            prev_parities = parities
            print(f"{a}/{denom}")
        print("-------")

if __name__ == "__main__":
    # print(is_reduced(2,2))
    main()

