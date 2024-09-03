"""
The solver for the question:\n

Four students play Guandan. They use two packs of cards. Each pack of cards contains 54 different cards.
Each student receives 27 cards at the beginning of the game.
Without considering the position of the students (i.e., we only consider the way of splitting these 108 cards into 4 group evenly),
find the number of ways to distribute these cards.

"""

from problem1 import comb_of_question1
from problem2 import comb_of_question2
from typing import Tuple, Iterable

import math

def solutions_of_m1m2() -> Iterable[Tuple[int, int]]:
    a = 54
    while a >= 0:
        yield (a, (54 - a) // 2)
        a -= 2

if __name__ == "__main__":
    M12 = solutions_of_m1m2()
    result = 0
    counter = 0
    
    for data in M12:
        counter += 1
        M1, M2 = data
        temp1 = 0
        card_choose_comb = math.comb(54, M1) * math.comb(54 - M1, M2)
        if M1 == 0 and M2 > 0:
            temp1 = comb_of_question1(27, M2, 2)
        elif M1 > 0 and M2 == 0:
            temp1 = comb_of_question1(27, M1, 1)
        else:
            temp1 = comb_of_question2(27, M1, M2)
        result += card_choose_comb * (temp1 * temp1)
    
    assert counter == 28
    solution = "{:.3g}".format(result)
    
    print(f"There are {solution} ways to distribute these cards to 4 players!")
    