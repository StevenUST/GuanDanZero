"""
The solver for the question:\n

We have N identical balls and M different buckets. Now we put these balls into the bucket and no bucket contains more than K balls.\n
The bucket can be empty. Find the number of ways to distribute these balls.

"""

import math

def comb_of_question1(N : int, M : int, K : int) -> int:
    answer = 0
    
    for r in range(M):
        a = 1 if r % 2 == 0 else -1
        b = math.comb(M, r)
        if N < (K + 1) * r:
            continue
        else:
            c = math.comb((N - (K + 1) * r + M - 1), M - 1)
        answer += (a * b * c)
    
    return answer

if __name__ == "__main__":
    print("{:.3g}".format(comb_of_question1(27, 27, 2)))