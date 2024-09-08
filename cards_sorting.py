from typing import List, Iterable, Final
from functools import cmp_to_key

SUITS: Final[List[str]] = ['S', 'H', 'C', 'D']

POWERS: Final[List[str]] = ['2', '3', '4', '5', '6',
                            '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'B', 'R']

def card_comparator(card1 : str, card2 : str) -> int:
    if card1 == card2:
        return 0
    temp = (card1[1] in ['B', 'R'], card2[1] in ['B', 'R'])
    if temp[0] or temp[1]:
        if temp[0] and temp[1]:
            return 1 if card1[1] == 'R' else -1
        elif temp[0]:
            return 1
        else:
            return -1
    else:
        temp2 = (POWERS.index(card1[1]), POWERS.index(card2[1]))
        if temp2[0] < temp2[1]:
            return -1
        elif temp2[0] > temp2[1]:
            return 1
        else:
            temp3 = (SUITS.index(card1[0]), SUITS.index(card2[0]))
            if temp3[0] < temp3[1]:
                return -1
            else:
                return 1
        

class CardsSorter:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def sorting_cards(cards : List) -> None:
        list.sort(cards, key=cmp_to_key(card_comparator))

if __name__ == "__main__":
    a = ('H2', 'D2', 'H3', 'S4', 'D4', 'SB', 'SB', 'HR')
    b = list(a)
    from random import shuffle
    
    shuffle(b)
    
    print(b)
    
    CardsSorter.sorting_cards(b)
    
    print(b)