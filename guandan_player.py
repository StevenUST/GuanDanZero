from typing import List, Optional

from utils import *

class GDPlayer(object):
    
    def __init__(self, model : object, player_pos : int) -> None:
        self.model = model
        self.pos = player_pos