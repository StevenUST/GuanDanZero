"""
The solver for the question:\n

We have N identical balls and M different buckets. Now we put these balls into the bucket.
In these buckets, M1 of them can contain at most 1 ball. The remaining M2 of them can ocntain at most 2 balls.
Find the number of ways to distribute these balls.

"""

import math

def comb_of_question2(N : int, M1 : int, M2 : int) -> int:
    M = M1 + M2
    answer = math.comb(N + M - 1, M - 1)
    
    for b in range(1, M + 1):
        c = 1 if b % 2 == 1 else -1
        temp = 0
        for a in range(0, b + 1):
            c1 = math.comb(M1, a)
            c2 = math.comb(M2, b - a)
            temp2 = N + a - 3 * b
            if temp2 < 0:
                continue
            else:
                c3 = math.comb(temp2 + M - 1, M - 1)
            temp += c1 * c2 * c3
        answer -= c * temp
    
    return answer

if __name__ == "__main__":
    print("{:.4g}".format(comb_of_question2(27, 50, 2)))