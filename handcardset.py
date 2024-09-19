from typing import Dict, List, Union, Optional, Final, Tuple

from utils import getWildCard, isLegalCard, addCardToDict as ACTD, getCardDict
from utils3 import getCardDictWithoutSuit, findAllCombWithoutSuit
from cardcomb import CardCombNoSuit, CardComb

CardNumToNum: Final[Dict[str, int]] = {
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, 'B': 15, 'R': 16
}

class HandCardSet(object):
    '''
    Only suitable for card set with size less than 6.
    '''
    
    def __init__(self, cards : List[str], level : int, consider_suit : bool) -> None:
        self.consider_suit : bool = consider_suit
        self.level = level
        self.combs : Optional[Union[List[CardComb], List[CardCombNoSuit]]] = None
        if consider_suit:
            self.card_dict = getCardDict()
            self.is_sf : Optional[Tuple[bool, int]] = None
            self.addCardToDict(cards)
        else:
            self.card_dict = getCardDictWithoutSuit()
            self.addCardToDictWithoutSuit(self.card_dict, cards, level)
            self.combs = findAllCombWithoutSuit(cards, self.card_dict, level)
            self.is_sf : Optional[Tuple[bool, int]] = None
            for comb in self.combs:
                if comb.t == 'StraightFlush':
                    self.is_sf = (True, comb.rank)
                    break
            if self.is_sf is None:
                self.is_sf = (False, -1)
    
    def addCardToDict(self, cards : List[str]) -> None:
        ACTD(self.card_dict, cards)
    
    def addCardToDictWithoutSuit(self, cards : List[str]) -> None:
        wild_card = getWildCard(self.level)
        for card in cards:
            if isLegalCard(card):
                if card == wild_card:
                    self.card_dict[16] += 1
                elif card == 'HR':
                    self.card_dict[15] += 1
                elif card == 'SB':
                    self.card_dict[14] += 1
                else:
                    num = card[1]
                    self.card_dict[num] += 1
                self.card_dict[17] += 1
            else:
                print(f"Illegal card \"{card}\" is found!")

    def play_action(self, action : CardCombNoSuit)