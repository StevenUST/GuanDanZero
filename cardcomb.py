from typing import Iterable, Final, Tuple, Dict

NumToCardNum: Final[Dict[int, str]] = {
    0: 'PASS', 1: 'A', 2: '2', 3: '3', 4 : '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: 'B', 16: 'R'
}

class CardComb:
    
    def __init__(self, ctype : str, crank : int, cards : Iterable[str]) -> None:
        self.t : str = ctype
        # From 1 to 16. For Joker Bomb, it is 16.
        if self.t == 'PASS':
            self.rank : int = 0
            self.cards : Tuple = ('PASS',)
        else:
            self.rank : int = crank
            self.cards : Tuple = tuple(cards)
    
    @staticmethod
    def create_pass_cardcomb() -> object:
        return CardComb('PASS', 0, [])
    
    def __str__(self) -> str:
        if self.t == 'PASS':
            return str(['PASS', 'PASS', 'PASS'])
        rank = None
        if self.t in ['ThreePairs', 'TwoTrips', 'Straight', 'StraightFlush']:
            rank = NumToCardNum[self.rank]
        else:
            rank = NumToCardNum[self.rank + 1]
        return str([self.t, rank, self.cards])
    
    def is_pass(self) -> bool:
        return self.t == 'PASS'

    def __hash__(self) -> int:
        return hash(self.cards)
    
    def __eq__(self, other : object) -> bool:
        if not isinstance(other, CardComb):
            return False
        return self.t == other.t and self.rank == other.rank and self.cards == other.cards