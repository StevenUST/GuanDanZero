from typing import Iterable, Final, Tuple, Dict, Optional

NumToCardNum: Final[Dict[int, str]] = {
    0: 'PASS', 1: 'A', 2: '2', 3: '3', 4 : '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: 'B', 16: 'R'
}

def isLargerThanRank(r1: int, r2: int, level: Optional[int]) -> bool:
    '''
    return @param r1 > @param r2 base on @param level
    '''
    if level is None or r1 == 0 or r2 == 0:
        return r1 > r2
    else:
        t = [i for i in range(1, 14)]
        if level != 13:
            _ = t.pop(level - 1)
            t.insert(12, level)
        t.extend([14, 15, 16])
        return t.index(r1) > t.index(r2)

class CombBase:
    
    def __init__(self, ctype : str, crank : int) -> None:
        self.t : str = ctype
        if self.t == 'PASS':
            self.rank : int = 0
        else:
            self.rank : int = crank
    
    @staticmethod
    def pass_comb() -> object:
        return CombBase('PASS', 0)
    
    def __str__(self) -> str:
        if self.t == 'PASS':
            return str(['PASS', 0])
        rank = None
        if self.t in ['ThreePairs', 'TwoTrips', 'Straight', 'StraightFlush']:
            rank = NumToCardNum[self.rank]
        else:
            rank = NumToCardNum[self.rank + 1]
        return str([self.t, rank])
    
    def is_pass(self) -> bool:
        return self.t == 'PASS'

    def __hash__(self) -> int:
        return hash(f"{self.t}, {self.rank}")
    
    def __eq__(self, other : object) -> bool:
        if not isinstance(other, CombBase):
            return False
        return self.t == other.t and self.rank == other.rank

class CardComb(CombBase):
    
    def __init__(self, ctype : str, crank : int, cards : Iterable[str]) -> None:
        super().__init__(ctype, crank)
        # From 1 to 16. For Joker Bomb, it is 16.
        if self.t == 'PASS':
            self.cards : Tuple = ('PASS',)
        else:
            self.cards : Tuple = tuple(cards)
    
    @staticmethod
    def pass_cardcomb() -> object:
        return CardComb('PASS', 0, [])
    
    def num_wild_card(self, wild_card : str) -> int:
        return self.cards.count(wild_card)
    
    @staticmethod
    def actionComparision(action : object, base : object, level : int) -> bool:
        '''
        If @param action can be played given the last action is @param base, it returns True; Else it returns False.
        '''
        if not (isinstance(action, CardComb) and isinstance(base, CardComb)):
            raise ValueError("@param action and @param base both must be instances of @class CombBase!")
        if base.is_pass():
            return not action.is_pass()
        if base.t == 'Bomb' and base.rank == 16:
            return False
        if base.t in ['Bomb', 'StraightFlush']:
            if not action.t in ['Bomb', 'StraightFlush']:
                return False
            if base.t == 'Bomb':
                if action.t == 'Bomb':
                    if len(action.cards) < len(base.cards):
                        return False
                    elif len(action.cards) == len(base.cards):
                        return isLargerThanRank(action.rank, base.rank, level)
                    else:
                        return True
                else:
                    return len(base.cards) < 6
            else:
                if action.t == 'Bomb':
                    return len(action.cards) >= 6
                else:
                    return isLargerThanRank(action.rank, base.rank, None)
        else:
            if action.t in ['Bomb', 'StraightFlush']:
                return True
            elif action.t != base.t:
                return False
            else:
                if base.t in ['ThreePairs', 'TwoTrips', 'Straight']:
                    return isLargerThanRank(action.rank, base.rank, None)
                else:
                    return isLargerThanRank(action.rank, base.rank, level)
    
    def __str__(self) -> str:
        if self.t == 'PASS':
            return str(['PASS', 'PASS', 'PASS'])
        rank = None
        if self.t in ['ThreePairs', 'TwoTrips', 'Straight', 'StraightFlush']:
            rank = NumToCardNum[self.rank]
        else:
            rank = NumToCardNum[self.rank + 1]
        return str([self.t, rank, self.cards])

    def __hash__(self) -> int:
        return hash(self.cards)
    
    def __eq__(self, other : object) -> bool:
        if not isinstance(other, CardComb):
            return False
        return self.t == other.t and self.rank == other.rank and self.cards == other.cards