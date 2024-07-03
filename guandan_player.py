from typing import List, Optional

from utils import *

class GDPlayer(object):
    
    def __init__(self, model : object, player_pos : int, perfect_info : bool) -> None:
        self.model = model
        self.card_dict = dict()
        self.my_pos : int = player_pos
        self.my_last_action : List = list()
        self.greater_pos : int = -1
        self.current_greatest_action : Optional[List[str]] = None
        self.levels = [1, 1, 1] # [Current Level, My Level, Opponent Level]
        
        if perfect_info:
            self.oppo_dicts = list()
            for _ in range(3):
                self.oppo_dicts.append(dict())
        else:
            self.oppo_dicts = dict()
        self.perfect_info = perfect_info

    def receive_cards(self, cards : List[str]) -> None:
        self.card_dict = cardsToDict(cards)
        self.card_combs = findAllCombs(self.card_dict, self.levels[0])

    def receive_oppo_cards(self, cards : List[str], index : int) -> None:
        if self.perfect_info:
            self.oppo_dicts[self.get_relative_index_of_player(index)] = cardsToDict(cards)
        else:
            self.oppo_dicts = cardsToDict(cards)

    def update_after_action(self, oppo_action : Optional[List], index : int) -> None:
        if oppo_action is not None:
            self.greater_pos = index
            self.current_greatest_action = oppo_action.copy()

    def update_self_card_combs(self, action : Optional[List]) -> bool:
        if action is not None:
            self.greater_pos = self.my_pos
            self.current_greatest_action = action.copy()
        self.my_last_action = action.copy()
        return updateCardCombsAfterAction(self.card_combs, self.card_dict, action)

    def update_after_small_episode_over(self, score : int) -> None:
        if score > 0:
            self.levels[0] = min(13, self.levels[0] + score)
            self.levels[1] = min(13, self.levels[1] + score)
        else:
            self.levels[0] = min(13, self.levels[0] + score)
            self.levels[2] = min(13, self.levels[2] + score)

    def update_after_game_over(self) -> None:
        for i in range(3):
            self.levels[i] = 1
        self.card_dict = dict()
        self.my_last_action = list()
        self.greater_pos = -1
        self.current_greatest_action = None

    def get_relative_index_of_player(self, index : int) -> int:
        if index == self.my_pos:
            return -1
        elif index > self.my_pos:
            return index - self.my_pos - 1
        else:
            return index + 3 - self.my_pos