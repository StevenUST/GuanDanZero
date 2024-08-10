from typing import List, Dict, Optional, Tuple

from cardcomb import CardComb
from utils import cardsToDict, findAllCombs, filterActions, getWildCard, updateCardCombsAfterAction

# Need to be removed later
def dummy() -> CardComb:
    return CardComb.create_pass_cardcomb()

class GDPlayer:
    
    def __init__(self, id: int) -> None:
        self.cards : Optional[Dict[str, int]] = None
        self.combs : Optional[List[CardComb]] = None
        self.level : int = 0
        self.id = id
    
    def init_cards(self, cards : List[str], level : int) -> None:
        self.cards = cardsToDict(cards)
        self.level = level
        self.combs = findAllCombs(self.cards, level)

    def get_all_legal_actions(self, current_greatest_action : Optional[CardComb] = None) -> List[CardComb]:
        '''
        If you want to get all actions, just call this function without any input parameter.
        '''
        return filterActions(self.combs, current_greatest_action, self.level)

    def get_action(self, oppo_cards : Dict[str, int], last_action : Optional[CardComb]) -> CardComb:
        my_action = None
        if last_action is None or last_action.is_pass():
            my_action = dummy()
        else:
            legal_actions : List[CardComb] = filterActions(self.combs, last_action, self.level)
            my_action = dummy()
        self.combs = updateCardCombsAfterAction(self.combs, self.cards, my_action)
        return my_action
    
    def no_card_left(self) -> bool:
        return self.cards['total'] == 0
    
    def stage_of_the_player(self) -> int:
        '''
        Stages: (stage_num : interval of handcards number)\n
        0 : [20, 27]\n
        1 : [13, 19]\n
        2 : [7, 12]\n
        3 : [6, 6]\n
        4 : [5, 5]\n
        ...\n
        7 : [2, 2]\n
        8 : [1, 1]\n
        '''
        num = self.cards['total']
        if num >= 20:
            return 0
        elif num >= 13:
            return 1
        elif num >= 7:
            return 2
        else:
            return 9 - num

class GDGame2P:
    
    def __init__(self):
        self.p1 : GDPlayer = GDPlayer(1)
        self.p2 : GDPlayer = GDPlayer(2)

    def init_players(self, cards1 : List[str], cards2 : List[str], level : int) -> None:
        self.p1.init_cards(cards1, level)
        self.p2.init_cards(cards2, level)
    
    def play(self) -> int:
        greatest_action : Optional[CardComb] = None
        current_player_id = 1
        while self.game_over() == 0:
            if current_player_id == 1:
                greatest_action = self.p1.get_action(self.p2.cards, greatest_action)
            else:
                greatest_action = self.p2.get_action(self.p1.cards, greatest_action)
        return self.game_over()
    
    def game_over(self) -> int:
        if self.p1.no_card_left():
            return self.p1.id
        if self.p2.no_card_left():
            return self.p2.id
        return 0