from typing import List, Dict, Optional, Tuple

from cardcomb import CardComb, CombBase
from utils import cardsToDict, findAllCombs, filterActions, getWildCard, updateCardCombsAfterAction, generateNRandomCardLists, getChoiceUnderAction,\
    checkCardCombTypeRank

from random import randint

class Cards(object):

    def __init__(self):
        self.cd1 : Optional[Dict[str, int]] = None
        self.cd2 : Optional[Dict[str, int]] = None
        self.comb1 : Optional[List[CardComb]] = None
        self.comb2 : Optional[List[CardComb]] = None
        self.last_moves : Optional[CardComb] = None
        self.level = 1

    def init_cards(self, card1: List[str], card2 : List[str], level: int) -> None:
        self.level = level
        self.cd1 = cardsToDict(card1)
        self.cd2 = cardsToDict(card2)
        self.comb1 = findAllCombs(self.cd1, self.level)
        self.comb2 = findAllCombs(self.cd2, self.level)
        self.last_move : Optional[CardComb] = None
        # Either 1 or 2.
        self.current_player : int = 1

    def current_state(self, current_player : bool) -> List:
        if current_player:
            pass
        else:
            pass

    def get_action(self) -> Tuple[int, CombBase, List[float]]:
        pass
    
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

    def __init__(self):
        self.cards: Cards = Cards()

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

    def start_self_play_dummy(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        pass
        # self.init_players()  # 需要随机自动分配牌
        # p1 = self.p1
        # p2 = self.p2
        # states, mcts_probs, current_players = [], [], []
        # while True:
        #     move, move_probs = player.get_action(self.board,
        #                                          temp=temp,
        #                                          return_prob=1)
        #     # store the data
        #     states.append(self.board.current_cards())  # 两个人的手牌
        #     mcts_probs.append(move_probs)
        #     current_players.append(self.board.current_player)
        #     # perform a move
        #     self.cards.do_move(move)
        #     # if is_shown:
        #     #    self.graphic(self.board, p1, p2)
        #     end, winner = self.Cards.game_end()
        #     if end:
        #         # winner from the perspective of the current player of each state
        #         winners_z = np.zeros(len(current_players))
        #         if winner != -1:
        #             winners_z[np.array(current_players) == winner] = 1.0
        #             winners_z[np.array(current_players) != winner] = -1.0
        #         # reset MCTS root node
        #         player.reset_player()
        #         if is_shown:
        #             if winner != -1:
        #                 print("Game end. Winner is player:", winner)
        #             else:
        #                 print("Game end. Tie")
        #         return winner, zip(states, mcts_probs, winners_z)
