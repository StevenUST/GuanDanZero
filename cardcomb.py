from typing import Iterable, Final, Tuple, Dict, Optional

NumToCardNum: Final[Dict[int, str]] = {
    0: 'PASS', 1: 'A', 2: '2', 3: '3', 4 : '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: 'B', 16: 'R'
}

TypeIndex : Final[Dict[str, int]] = {'Single' : 0, 'Pair' : 1, 'Trip' : 2, 'ThreeWithTwo' : 3, 
                                     'TwoTrips' : 4, 'ThreePairs' : 5, 'Straight' : 6, 'StraightFlush' : 7,
                                     'Bomb' : 8}

def isLargerThanRank(r1: int, r2: int, level: Optional[int]) -> bool:
    '''
    return @param r1 > @param r2 base on @param level
    '''
    if r1 == r2:
        return False
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
    
    def __init__(self, ctype : str, crank : int, num_cards : int) -> None:
        self.t : str = ctype
        if self.t == 'PASS':
            self.rank : int = 0
            self.cards_num : int = 0
        else:
            self.rank : int = crank
            self.cards_num : int = num_cards
    
    @staticmethod
    def pass_comb() -> object:
        return CombBase('PASS', 0, 0)
    
    @staticmethod
    def jokerbomb_comb() -> object:
        return CombBase('Bomb', 16, 4)
    
    @staticmethod
    def actionComparision(action : Optional[object], base : Optional[object], level : int) -> bool:
        '''
        If @param action can be played given the last action is @param base, it returns True; Else it returns False.
        '''
        if base is None or base.is_pass():
            return action is not None and not action.is_pass()
        if action is None or action.is_pass():
            return base is not None and not base.is_pass()
        if base.is_joker_bomb() or action.is_joker_bomb():
            if (base.rank == action.rank) or base.rank == 16:
                return False
            else:
                return True
        if base.t in ['Bomb', 'StraightFlush']:
            if not action.t in ['Bomb', 'StraightFlush']:
                return False
            if base.t == 'Bomb':
                if action.t == 'Bomb':
                    if action.cards_num < base.cards_num:
                        return False
                    elif action.cards_num == base.cards_num:
                        return isLargerThanRank(action.rank, base.rank, level)
                    else:
                        return True
                else:
                    return base.cards_num < 6
            else:
                if action.t == 'Bomb':
                    return action.cards_num >= 6
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
    
    def type_index(self) -> int:
        if self.is_pass():
            return 0
        elif self.t == 'Bomb' and self.rank == 16:
            return 16
        else:
            index = TypeIndex[self.t]
            if index == 8:
                index += (self.cards_num - 4)
            return index
    
    def __str__(self) -> str:
        if self.is_pass():
            return str(['PASS', 0, 0])
        elif self.is_joker_bomb():
            return str(['Bomb', 'JOKER', 4])
        rank = None
        if self.t in ['ThreePairs', 'TwoTrips', 'Straight', 'StraightFlush']:
            rank = NumToCardNum[self.rank]
        else:
            rank = NumToCardNum[self.rank + 1]
        return str([self.t, rank, self.cards_num])
    
    def is_pass(self) -> bool:
        return self.t == 'PASS'
    
    def is_joker_bomb(self) -> bool:
        return self.t == 'Bomb' and self.rank == 16

    def __hash__(self) -> int:
        return hash(f"{self.t}, {self.rank}")
    
    def __eq__(self, other : object) -> bool:
        if not isinstance(other, CombBase):
            return False
        return self.t == other.t and self.rank == other.rank and self.cards_num == other.cards_num

class CardComb(CombBase):
    
    def __init__(self, ctype : str, crank : int, cards : Iterable[str]) -> None:
        super().__init__(ctype, crank, len(cards))
        # From 1 to 16. For Joker Bomb, it is 16.
        if self.t == 'PASS':
            self.cards : Tuple = ('PASS',)
        else:
            self.cards : Tuple = tuple(cards)
    
    @staticmethod
    def pass_cardcomb() -> object:
        return CardComb('PASS', 0, [])
    
    @staticmethod
    def jokerbomb_cardcomb() -> object:
        return CardComb('Bomb', 16, ['SB', 'SB', 'HR', 'HR'])
    
    def num_wild_card(self, wild_card : str) -> int:
        return self.cards.count(wild_card)
    
    def to_comb_base(self) -> CombBase:
        return CombBase(self.t, self.rank, self.cards_num)
    
    def __str__(self) -> str:
        if self.is_pass():
            return str(['PASS', 'PASS', 'PASS'])
        elif self.is_joker_bomb():
            return str(['Bomb', 'JOKER', str(self.cards)])
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