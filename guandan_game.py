from typing import List, Dict, Optional, Tuple, Callable

from cardcomb import CardComb, CombBase
from utils import cardsToDict, findAllCombs, getWildCard, updateCardCombsAfterAction, generateNRandomCardLists, getChoiceUnderAction, getChoiceUnderThreeWithTwo,\
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
        self.update_comb_index(self.comb1, None, 1)
        self.update_comb_index(self.comb2, None, 2)
    
    def current_player_comb_indices(self) -> List[int]:
        if self.current_player == 1:
            return self.comb_index1
        else:
            return self.comb_index2

    def update_comb_index(self, combs : List[CardComb], leave_indices : Optional[List[int]], player_index : int, remove : bool = False) -> None:
        val = 0 if remove else 1
        if player_index == 1:
            for comb in combs:
                index = indexOfCombBase(comb)
                if leave_indices is None or not index in leave_indices:
                    self.comb_index1[index] = val
        else:
            for comb in combs:
                index = indexOfCombBase(comb)
                if leave_indices is None or not index in leave_indices:
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
        
        return t1, t2
    
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
        
        return t1, t2
        
    def level_list(self) -> nplist:
        answer = [0] * 13
        answer[self.level - 1] = 1
        t = ascontiguousarray(answer, dtype=npf32)
        return t
    
    def do_action(self, action : CardComb):
        self.last_move = action
        fail_combs = None
        if self.current_player == 1:
            fail_combs, self.comb1 = updateCardCombsAfterAction(self.comb1, self.cd1, action)
            successful_comb_indices = [indexOfCombBase(c) for c in self.comb1]
            self.update_comb_index(fail_combs, successful_comb_indices, 1, True)
            self.current_player = 2
        else:
            fail_combs, self.comb2 = updateCardCombsAfterAction(self.comb2, self.cd2, action)
            successful_comb_indices = [indexOfCombBase(c) for c in self.comb2]
            self.update_comb_index(fail_combs, successful_comb_indices, 2, True)
            self.current_player = 1

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

    def __init__(self):
        self.cards: Cards = Cards()

    def init_players(self, cards1: List[str], cards2: List[str], level: int) -> None:
        self.cards.init_cards(cards1, cards2, level)

    def start_self_play(self, player : MCTSPlayer) -> Tuple[List[List[nplist]], List[nplist], List[int], int, int]:
        random_cards = generateNRandomCardLists((12, 12))
        level = randint(1, 13)
        self.init_players(random_cards[0], random_cards[1], level)
        states, mcts_probs, current_players = [], [], []
        data_count = 0
        while not self.game_over():
            action_base, action_probs = player.get_action(self.cards)

            # Store data
            my_states = self.cards.current_states(True)
            oppo_states = self.cards.current_states(False)
            last_action = self.cards.last_move_list()
            current_level = self.cards.level_list()
            
            states.append([my_states[0], reshape(my_states[1], (16, 15, 1)), oppo_states[0], reshape(oppo_states[1], (16, 15, 1)), last_action[0], last_action[1], current_level])
            mcts_probs.append(action_probs)
            current_players.append(self.cards.current_player)
            data_count += 1
            
            # Filter All choices
            player_data = self.cards.get_player_status(self.cards.current_player)
            choices : List[CardComb] = list(filter(lambda comb : checkCardCombTypeRank(comb, action_base), player_data[1]))
            choice : Optional[CardComb] = None
            
            # Get the choice under the given action
            if action_base.t == "ThreeWithTwo":
                choice : CardComb = getChoiceUnderThreeWithTwo(player_data[0], player_data[1], choices, level)
            else:
                choice : CardComb = getChoiceUnderAction(player_data[0], choices, getWildCard(level))
            self.cards.do_action(choice)
        player.reset_player()
        return (states, mcts_probs, current_players, self.cards.has_a_winner(), data_count)

    def start_play_against_other(self, player : MCTSPlayer, baseline_player : MCTSPlayer) -> int:
        random_cards = generateNRandomCardLists((2, 2))
        level = randint(1, 1)
        self.init_players(random_cards[0], random_cards[1], level)
        action_base : Optional[CombBase] = None
        
        while not self.game_over():
            print(f"game_over = {self.game_over()}")
            if self.cards.current_player == 1:
                action_base = player.get_action(self.cards, need_prob=False)
            else:
                action_base = baseline_player.get_action(self.cards, need_prob=False)
            
            player_data = self.cards.get_player_status(self.cards.current_player)
            choices : List[CardComb] = list(filter(lambda comb : checkCardCombTypeRank(comb, action_base), player_data[1]))
            choice : Optional[CardComb] = None
            
            # Get the choice under the given action
            if action_base.t == "ThreeWithTwo":
                choice : CardComb = getChoiceUnderThreeWithTwo(player_data[0], player_data[1], choices, level)
            else:
                choice : CardComb = getChoiceUnderAction(player_data[0], choices, getWildCard(level))
            self.cards.do_action(choice)
        player.reset_player()
        baseline_player.reset_player()
        return 1 if self.cards.has_a_winner() == 1 else -1

    def game_over(self) -> bool:
        return self.cards.has_a_winner() > 0
