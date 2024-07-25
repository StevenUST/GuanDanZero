from typing import List, Set, Dict, Optional

from utils import isSameAction

import collections

def dict_to_str(d : Dict) -> str:
    temp = sorted(d)
    
    temp2 = list()
    for key in temp:
        temp2.append(f"{key}:{d[key]}")
    
    return str(temp2)

def same_dicts(d1 : Dict, d2 : Dict) -> bool:
    if not d1:
        return not d2
    elif not d2:
        return not d1
    
    keys1 = list(d1.keys())
    keys2 = list(d2.keys())
    
    if collections.Counter(keys1) != collections.Counter(keys2):
        return False
    
    keys = keys1.copy()
    
    for key in keys:
        if d1[key] != d2[key]:
            return False
    
    return True

class GDNode(object):
    
    def __init__(self, index : int, card_dict1 : Dict[str, int], card_dict2 : Dict[str, int], layer : int, level : int, greatest_action : Optional[List] = None):
        self.player_index = index
        self.card_dict1 = card_dict1
        self.card_dict2 = card_dict2
        self.layer_num = layer
        self.level = level
        self.current_greatest_action : Optional[List] = greatest_action
        self.parent : Optional[Set] = None
        self.children : Optional[List] = None
        self.actions : Optional[List] = None
        self.best_child_index : Optional[Set] = None
        self.reward : int = -100

    @staticmethod
    def create_dummy_node(index : int, card_dict1 : Dict[str, int], card_dict2 : Dict[str, int], layer : int, level : int, greatest_action : Optional[List]) -> object:
        g : Optional[List] = None
        if greatest_action is not None:
            g = greatest_action.copy()
        return GDNode(index, card_dict1.copy(), card_dict2.copy(), layer, level, g)

    def add_child_node(self, child : object, action : List) -> None:
        if isinstance(child, GDNode):
            if self.children is None:
                self.children = list()
                self.actions = list()
                self.children.append(child)
                self.actions.append(action)
            elif len(self.children) == 0 or not any(c == child for c in self.children):
                self.children.append(child)
                self.actions.append(action)
    
    def remove_child_node(self, child : object) -> None:
        if self.children is None:
            return False
        if isinstance(child, GDNode):
            index = -1
            try:
                index = self.children.index(child)
            except ValueError:
                pass
            if index == -1:
                return
            self.children.remove(child)
            _ = self.actions.pop(index)
            
            if len(self.children) == 0:
                self.children = None
                self.actions = None
            
            child.remove_parent(self, False)
    
    def add_parent(self, p : object) -> None:
        if isinstance(p, GDNode):
            if self.parent is None:
                self.parent = set()
            if not p in self.parent:
                self.parent.add(p)
    
    def remove_parent(self, p : object, remove_child : bool = True) -> None:
        if isinstance(p, GDNode):
            if self.parent == None:
                return
            if p in self.parent:
                self.parent.remove(p)
                if remove_child:
                    p.remove_child_node(self)
            if len(self.parent) == 0:
                self.parent = None
    
    def reward_back_track(self, reward : int, best_child : Optional[object]) -> None:
        if best_child is not None and reward == 1:
            index : int = self.children.index(best_child)
            if self.best_child_index is None:
                self.best_child_index = set()
            self.best_child_index.add(index)
        
        self.reward = max(self.reward, reward)
        
        if self.parent is None or len(self.parent) == 0:
            return
        else:
            for p in self.parent:
                if isinstance(p, GDNode):
                    p.reward_back_track(reward, \
                        GDNode.create_dummy_node(self.player_index, self.card_dict1, self.card_dict2, self.layer_num, self.level, self.current_greatest_action))
    
    def set_reward(self, reward : int) -> None:
        self.reward = reward

    def is_leaf(self) -> bool:
        return self.children is None or len(self.children) == 0

    def __str__(self) -> str:
        answer = f"{self.player_index}, "
        answer += str(self.card_dict1)
        answer += ', '
        answer += str(self.card_dict2)
        
        return answer

    def __hash__(self) -> int:
        return hash(dict_to_str(self.card_dict1) + dict_to_str(self.card_dict2))

    def __eq__(self, other : object) -> bool:
        if not isinstance(other, GDNode):
            return False
        return self.player_index == other.player_index and same_dicts(self.card_dict1, other.card_dict1) and same_dicts(self.card_dict2, other.card_dict2) \
            and self.layer_num == other.layer_num and self.level == other.level and isSameAction(self.current_greatest_action, other.current_greatest_action)