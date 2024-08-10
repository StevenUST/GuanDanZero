from typing import List, Dict, Optional, Tuple

from cardcomb import CardComb
from utils import cardsToDict, findAllCombs, filterActions, getWildCard, updateCardCombsAfterAction

# Need to be removed later


def dummy() -> CardComb:
    return CardComb.create_pass_cardcomb()


class Cards(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.states = {}
        # need how many pieces in a row to win
        self.players = [1, 2]  # player1 and player2

    def init_cards(self, cards: List[str], level: int, start_player=0):
        self.current_player = self.players[start_player]  # start player
        self.level = level
        # keep available moves in a list
        self.combs_for_model = findAllCombs(
            self.cards, level)   # 这里轮到谁出牌就算谁可行的所有action
        self.combs_with_suit =  # 这里是有级牌的手牌
        self.states = {}
        self.last_move = -1

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        # 这里是模型的输入

    def do_move(self, move):
        self.states[move] = self.current_player
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        # 这里需要更新手牌，更新可行的出牌
        self.last_move = move

    def has_a_winner(self):
        # 返回两个值，是否赢(1 / 0), 赢的人是谁（1 / 2）

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class GDGame2P:

    def __init__(self, turn):
        self.p1: GDPlayer = GDPlayer(1)
        self.p2: GDPlayer = GDPlayer(2)
        self.turn = turn

    def init_players(self, cards1: List[str], cards2: List[str], level: int) -> None:
        self.p1.init_cards(cards1, level)
        self.p2.init_cards(cards2, level)

    def play(self) -> int:
        greatest_action: Optional[CardComb] = None
        current_player_id = 1
        while self.game_over() == 0:
            if current_player_id == 1:
                greatest_action = self.p1.get_action(
                    self.p2.cards, greatest_action)
            else:
                greatest_action = self.p2.get_action(
                    self.p1.cards, greatest_action)
        return self.game_over()

    def game_over(self) -> int:
        if self.p1.no_card_left():
            return self.p1.id
        if self.p2.no_card_left():
            return self.p2.id
        return 0

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.init_players()  # 需要随机自动分配牌
        p1 = self.p1
        p2 = self.p2
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_cards())  # 两个人的手牌
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.cards.do_move(move)
            # if is_shown:
            #    self.graphic(self.board, p1, p2)
            end, winner = self.Cards.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
