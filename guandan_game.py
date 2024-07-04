from typing import List, Union, Tuple, Iterable

from guandan_player import GDPlayer

from utils import inRange, generateNRandomCardLists
from numpy import random

class GDGame(object):
    
    def __init__(self, players : Union[List[GDPlayer], Tuple[GDPlayer]]) -> None:
        assert isinstance(players, Iterable)

        self.players = list(players)
        self.players.sort(key = lambda p : p.my_pos)
        self.num_player = len(self.players)
        
        if self.num_player == 1:
            self.players[0].my_pos = 0
    
    def selfplay(self) -> None:
        '''
        This function play one round of game.\n
        Call this function repeatly to play multiple rounds of game.
        '''
        
        # Only allow one player to play
        if self.num_player != 1:
            return
        
        nums = tuple([random.randint(1, 28), random.randint(1, 28)])
        card_list = generateNRandomCardLists(nums)
        
        index = 0
        
        self.players[0].receive_cards(card_list[index])
        self.players[0].receive_oppo_cards(card_list[(index + 1) % 2], 1)
        
        greatest_index = 0
        greatest_action = None
        
        game_over = False
        
        while not game_over:
            action = self.players[0].make_action(greatest_action)
            
            if action is not None:
                greatest_index = index
                remaining = self.players[0].remove_cards(action[2])
                greatest_action = action.copy()
                card_list[index] = remaining.copy()
                if len(remaining) == 0:
                    game_over = True
            
            if game_over:
                break
            
            if greatest_index != index:
                greatest_action = None
            index = (index + 1) % 2
            self.players[0].receive_cards(card_list[index])
            self.players[0].receive_oppo_cards(card_list[(index + 1) % 2], 1)