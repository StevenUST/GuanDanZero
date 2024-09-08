from abc import abstractmethod
from typing import List, Set, Dict, Optional, Tuple

from cardcomb import CardComb

def dict_to_str(d : Dict) -> str:
    temp = sorted(d)
    
    temp2 = list()
    for key in temp:
        temp2.append(f"{key}:{d[key]}")
    
    return str(temp2)

class BaseNode(object):
    
    def __init__(self, index : int, n_index : int, layer : int, reward : int = -100):
        self.player_index = index
        self.node_index = n_index
        self.layer_num = layer
        self.reward = reward
        self.parent : Optional[Set[BaseNode]] = None
        self.children : Optional[List[BaseNode]] = None
    
    def set_reward(self, reward : int) -> None:
        self.reward = reward
    
    def is_leaf(self) -> bool:
        return self.children is None or len(self.children) == 0
    
    @abstractmethod
    def add_child_node(self, child : object, action : Optional[List] = None) -> None:
        raise NotImplementedError("This function is not implemented!")
    
    @abstractmethod
    def remove_child_node(self, child : object) -> None:
        raise NotImplementedError("This function is not implemented!")
    
    @abstractmethod
    def add_parent(self, p : object) -> None:
        raise NotImplementedError("This function is not implemented!")
    
    @abstractmethod
    def remove_parent(self, p : object, remove_child : bool = True) -> None:
        raise NotImplementedError("This function is not implemented!")
    
    def __hash__(self) -> int:
        raise NotImplementedError("This function is not implemented!")
    
    def __eq__(self, other : object) -> bool:
        raise NotImplementedError("This function is not implemented!")

class GDNode(BaseNode):
    
    def __init__(self, index : int, node_index : int, card_dict1 : Dict[str, int], card_dict2 : Dict[str, int], layer : int, level : int, greatest_action : Optional[CardComb] = None):
        super().__init__(index, node_index, layer)
        self.card_dict1 = card_dict1
        self.card_dict2 = card_dict2
        self.level = level
        self.current_greatest_action : Optional[CardComb] = greatest_action
        self.actions : Optional[List[CardComb]] = None
        # self.best_child_index : Optional[List] = None

    def add_child_node(self, child : object, action : Optional[CardComb] = None) -> None:
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
    
    def update_recursively(self) -> int:
        if self.reward in [1, -1]:
            return self.reward
        if self.is_leaf():
            self.reward = 1 if self.player_index == 1 else -1
            return self.reward
        if self.player_index == 0:
            r = -1000
            for c in self.children:
                if isinstance(c, GDNode):
                    r = max(r, c.update_recursively())
            self.reward = r
            return r
        else:
            r = 1000
            for c in self.children:
                if isinstance(c, GDNode):
                    r = min(r, c.update_recursively())
            self.reward = r
            return r

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
        return self.node_index == other.node_index

class GDResultNode(BaseNode):
    
    def __init__(self, player_index: int, node_index : int, layer: int, reward : int = -100):
        super().__init__(player_index, node_index, layer, reward)
    
    def add_child_node(self, child: object, action: Optional[List] = None) -> None:
        if not isinstance(child, GDResultNode):
            return
        if self.children is None:
            self.children = list()
        if len(self.children) == 0 or not any(c == child for c in self.children):
            self.children.append(child)
            if child.parent is None:
                child.parent = set()
            child.parent.add(self)
        if self.children is not None and len(self.children) == 0:
            self.children = None
    
    def remove_child_node(self, child: object) -> None:
        if not isinstance(child, GDResultNode):
            return
        if self.children is None:
            return
        if any(c == child for c in self.children):
            index = self.children.index(child)
            _ = self.children.pop(index)
            child.parent.remove(self)
    
    def add_parent(self, p: object) -> None:
        if not isinstance(p, GDResultNode):
            return
        if p.children is not None and any(c == self for c in p.children):
            return
        p.children.append(self)
        self.parent.add(p)
    
    def remove_parent(self, p: object, remove_child: bool = True) -> None:
        if not isinstance(p, GDResultNode):
            return
        if self.parent is None:
            return
        if p in self.parent:
            self.parent.remove(p)
            try:
                index = p.children.index(self)
                _ = p.children.pop(index)
            except:
                pass
    
    def __str__(self) -> str:
        return f"[Node {self.node_index}, player {self.player_index}, layer {self.layer_num}, reward = {self.reward}]"
    
    def __hash__(self) -> int:
        return hash(f"[{self.player_index}, {self.layer_num}]")
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GDResultNode):
            return False
        return self.node_index == other.node_index