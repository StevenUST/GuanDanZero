from typing import List, Union, Tuple, Iterable

from guandan_player import GDPlayer

from utils import inRange, generateNRandomCardLists, do_move
from numpy import random


class GDGame(object):

    def __init__(self, players: Union[List[GDPlayer], Tuple[GDPlayer]]) -> None:
        assert isinstance(players, Iterable) and inRange(len(players), (2, 4))

        self.players = list(players)
        self.players.sort(key=lambda p: p.my_pos)
        self.num_player = len(self.players)

    def has_a_winner(self) -> bool:
        for player in self.players:
            if player.has_no_cards():
                print(f"Player {player.my_pos} wins!")
                return True
        return False

    def self_play(self, player) -> None:
        '''
        This function play one round of game.\n
        Call this function repeatly to play multiple rounds of game.
        '''
        game_over = False

        random_numbers = tuple(random.randint(1, 28, size=self.num_player))
        card_lists = generateNRandomCardLists(random_numbers)

        for i in range(self.num_player):
            self.players[i].receive_cards(card_lists[i])

        player_index = [player.my_pos for player in self.players]
        idx = random.randint(1)
        states, mcts_probs, current_players = [], [], []

        while True:

            now = self.players[idx]
            # get_action需要写
            move, move_probs = player.get_action(card_lists, now,
                                                 temp=temp,
                                                 return_prob=1)

            states.append(card_lists)
            mcts_probs.append(move_probs)
            current_players.append(self.players[idx])

            # do_move需要写
            do_move(move, now)
            idx += 1
            if self.has_a_winner():
                break

        return current_players[-1], zip(states, mcts_probs, current_players)
