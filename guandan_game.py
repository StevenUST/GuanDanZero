from typing import List, Union, Tuple, Iterable

from guandan_player import GDPlayer

from utils import inRange, generateNRandomCardLists
from numpy import random

class GDGame(object):
    
    def __init__(self, players : Union[List[GDPlayer], Tuple[GDPlayer]]) -> None:
        assert isinstance(players, Iterable) and inRange(len(players), (2, 4))
        
        self.players = list(players)
        self.players.sort(key = lambda p : p.my_pos)
        self.num_player = len(self.players)
    
    def play(self) -> None:
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
        idx = random.randint(3)
        
        greatest_player_idx = -1
        greatest_action = None
        
        while not game_over:
            pass