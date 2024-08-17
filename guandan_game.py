from typing import List, Dict, Optional, Tuple, Callable

from cardcomb import CardComb
from utils import cardsToDict, findAllCombs, getWildCard, updateCardCombsAfterAction, generateNRandomCardLists, getChoiceUnderAction,\
    checkCardCombTypeRank, getFlagsForActions, cardDictToModelList, indexOfCombBase

from random import randint
from mcts_guandan import MCTSPlayer

from numpy import ndarray as nplist, ascontiguousarray, float32 as npf32, reshape

class Cards(object):

    def __init__(self):
        self.cd1 : Optional[Dict[str, int]] = None
        self.cd2 : Optional[Dict[str, int]] = None
        self.comb1 : Optional[List[CardComb]] = None
        self.comb2 : Optional[List[CardComb]] = None
        # What combination(s) the player can play freely.
        self.comb_index1 : List[int] = [0] * 194
        self.comb_index2 : List[int] = [0] * 194
        self.last_move : CardComb = CardComb.pass_cardcomb()
        self.level = 1
        # Either 1 or 2.
        self.current_player : int = 1

    def init_cards(self, card1: List[str], card2 : List[str], level: int) -> None:
        self.level = level
        self.cd1 = cardsToDict(card1)
        self.cd2 = cardsToDict(card2)
        self.comb1 = findAllCombs(self.cd1, self.level)
        self.comb2 = findAllCombs(self.cd2, self.level)
        self.update_comb_index(self.comb1, 1)
        self.update_comb_index(self.comb2, 2)
    
    def current_player_comb_indices(self) -> List[int]:
        if self.current_player == 1:
            return self.comb_index1
        else:
            return self.comb_index2

    def update_comb_index(self, combs : List[CardComb], player_index : int, remove : bool = False) -> None:
        val = 0 if remove else 1
        if player_index == 1:
            for comb in combs:
                index = indexOfCombBase(comb)
                self.comb_index1[index] = val
        else:
            for comb in combs:
                index = indexOfCombBase(comb)
                self.comb_index2[index] = val

    def current_states(self, current_player : bool) -> Tuple[nplist, nplist]:
        index = 0
        if current_player:
            index = self.current_player
        else:
            index = 3 - self.current_player
        
        hand_cards = None
        flags = list()
        
        if index == 1:
            hand_cards = cardDictToModelList(self.cd1, self.level)
            flags.append(getFlagsForActions(self.comb1, self.last_move, self.level, False))
        else:
            hand_cards = cardDictToModelList(self.cd2, self.level)
            flags.append(getFlagsForActions(self.comb2, self.last_move, self.level, False))
        
        t1 = ascontiguousarray(hand_cards, dtype=npf32)
        t2 = ascontiguousarray(flags, dtype=npf32)
        
        return (reshape(t1, (1, 70)), reshape(t2, (1, 16, 15, 1)))
    
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
        
        t1 = ascontiguousarray(action_list, dtype=npf32)
        t2 = ascontiguousarray(rank_list, dtype=npf32)
        
        return (reshape(t1, (1, 17)), reshape(t2, (1, 15)))
        
    def level_list(self) -> nplist:
        answer = [0] * 13
        answer[self.level - 1] = 1
        t = ascontiguousarray(answer, dtype=npf32)
        return reshape(t, (1, 13))
    
    def do_action(self, action : CardComb):
        self.last_move = action
        fail_combs = None
        if self.current_player == 1:
            fail_combs, self.comb1 = updateCardCombsAfterAction(self.comb1, self.cd1, action)
            self.update_comb_index(fail_combs, 1, True)
        else:
            fail_combs, self.comb2 = updateCardCombsAfterAction(self.comb2, self.cd2, action)
            self.update_comb_index(fail_combs, 2, True)
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

    def __init__(self, policy_value_fn : Callable):
        self.cards: Cards = Cards()
        self.p1 = MCTSPlayer(policy_value_fn, is_selfplay=1)

    def init_players(self, cards1: List[str], cards2: List[str], level: int) -> None:
        self.cards.init_cards(cards1, cards2, level)

    def start_self_play(self, player : MCTSPlayer) -> Tuple[List, List, List, int]:
        random_cards = generateNRandomCardLists((12, 12))
        level = randint(1, 13)
        self.init_players(random_cards[0], random_cards[1], level)
        states, mcts_probs, current_players = [], [], []
        while not self.game_over():
            action_base, action_probs = player.get_action(self.cards)
            
            # Store data
            my_states = self.cards.current_states(self.cards.current_player)
            oppo_states = self.cards.current_states(3 - self.cards.current_player)
            last_action = self.cards.last_move_list()
            current_level = self.cards.level_list()
            
            states.append([my_states[0], my_states[1], oppo_states[0], oppo_states[1], last_action[0], last_action[1], current_level])
            mcts_probs.append(action_probs)
            current_players.append(self.cards.current_player)
            
            # Filter All choices
            player_data = self.cards.get_player_status(self.cards.current_player)
            choices : List[CardComb] = list(filter(lambda comb : checkCardCombTypeRank(comb, action_base), player_data[1]))
            
            # Get the choice under the given action
            if action_base.t == "ThreeWithTwo":
                pass
            else:
                choice : CardComb = getChoiceUnderAction(player_data[0], choices, getWildCard(level))
            self.cards.do_action(choice)
        return (states, mcts_probs, current_players, self.cards.has_a_winner())

    def game_over(self) -> bool:
        return self.cards.has_a_winner() > 0
