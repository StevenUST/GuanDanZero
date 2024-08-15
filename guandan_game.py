from typing import List, Dict, Optional, Tuple, Callable

from cardcomb import CardComb, CombBase
from utils import cardsToDict, findAllCombs, getWildCard, updateCardCombsAfterAction, generateNRandomCardLists, getChoiceUnderAction,\
    checkCardCombTypeRank, getFlagsForActions, cardDictToModelList

from random import randint
from guandan_net_tensorflow import GuandanNetForTwo
from mcts_guandan import MCTS

from numpy import ndarray as nplist, ascontiguousarray, float32 as npf32

class Cards(object):

    def __init__(self, policy_value_fn : Callable):
        self.cd1 : Optional[Dict[str, int]] = None
        self.cd2 : Optional[Dict[str, int]] = None
        self.comb1 : Optional[List[CardComb]] = None
        self.comb2 : Optional[List[CardComb]] = None
        self.mcts1 : MCTS = MCTS(policy_value_fn, c_puct=5, n_playout=100)
        self.mcts2 : MCTS = MCTS(policy_value_fn, c_puct=5, n_playout=100)
        self.last_move : CardComb = CardComb.pass_cardcomb()
        self.level = 1

    def init_cards(self, card1: List[str], card2 : List[str], level: int) -> None:
        self.level = level
        self.cd1 = cardsToDict(card1)
        self.cd2 = cardsToDict(card2)
        self.comb1 = findAllCombs(self.cd1, self.level)
        self.comb2 = findAllCombs(self.cd2, self.level)
        # Either 1 or 2.
        self.current_player : int = 1

    def current_states(self, current_player : bool) -> Tuple[nplist, nplist]:
        index = 0
        if current_player:
            index = self.current_player
        else:
            index = 3 - self.current_player
        
        hand_cards = None
        flags = None
        
        if index == 1:
            hand_cards = cardDictToModelList(self.cd1, self.level)
            flags = getFlagsForActions(self.comb1, self.last_move, getWildCard(self.level))
        else:
            hand_cards = cardDictToModelList(self.cd2, self.level)
            flags = getFlagsForActions(self.comb2, self.last_move, getWildCard(self.level))
            
        return (ascontiguousarray(hand_cards, dtype=npf32), ascontiguousarray(flags, dtype=npf32))
        
    
    def last_move_list(self) -> Tuple[nplist, nplist]:
        action_list = [0] * 17
        rank_list = [0] * 15
        
        if self.last_move is None or self.last_move.is_pass():
            action_list[0] = 1
        else:
            type_index = self.last_move.type_index()
            action_list[type_index] = 1
            if type_index == 16:
                for i in range(15):
                    rank_list[i] = 1
            else:
                rank_list[self.last_move.rank - 1] = 1
        
        return (ascontiguousarray(action_list, dtype=npf32), ascontiguousarray(rank_list, dtype=npf32))
        
    def level_list(self) -> nplist:
        answer = [0] * 13
        answer[self.level - 1] = 1
        return ascontiguousarray(answer, dtype=npf32)

    def get_action(self) -> Tuple[int, List[int], List[float]]:
        if self.current_player == 1:
            acts, probs = self.mcts1.get_move_probs(self, temp=0.001)
        else:
            acts, probs = self.mcts2.get_move_probs(self, temp=0.001)
        return self.current_player, acts, probs
    
    def do_action(self, action : CardComb):
        self.last_move = action
        if self.current_player == 1:
            _, self.comb1 = updateCardCombsAfterAction(self.comb1, self.cd1, action)
        else:
            _, self.comb2 = updateCardCombsAfterAction(self.comb2, self.cd2, action)
        self.current_player = 3 - self.current_player

    def has_a_winner(self):
        if self.cd1['total'] == 0:
            return 1
        if self.cd2['total'] == 0:
            return 2
        return 0

    def get_current_player(self) -> int:
        return self.current_player

    def get_player_status(self, player_index : int) -> Tuple[Dict[str, int], List[CardComb]]:
        if player_index == 1:
            return (self.cd1, self.comb1)
        else:
            return (self.cd2, self.comb2)

class GDGame2P:

    def __init__(self, model_file : object = None):
        self.model : GuandanNetForTwo = GuandanNetForTwo(model_file=model_file)
        self.cards: Cards = Cards(model)

    def init_players(self, cards1: List[str], cards2: List[str], level: int) -> None:
        self.cards.init_cards(cards1, cards2, level)

    def start_self_play(self) -> List:
        random_cards = generateNRandomCardLists((12, 12))
        level = randint(1, 13)
        self.init_players(random_cards[0], random_cards[1], level)
        states, mcts_probs, current_players = [], [], []
        while not self.game_over():
            player_index, action, action_probs = self.cards.get_action()
            
            # Store data
            
            # Filter All choices
            player_data = self.cards.get_player_status(player_index)
            choices : List[CardComb] = list(filter(lambda comb : checkCardCombTypeRank(comb, action), player_data[1]))
            
            # Get the choice under the given action
            choice : CardComb = getChoiceUnderAction(player_data[0], choices, getWildCard(level))
            
            self.cards.do_action(choice)
            
        return self.cards.has_a_winner()

    def game_over(self) -> bool:
        return self.cards.has_a_winner() > 0
