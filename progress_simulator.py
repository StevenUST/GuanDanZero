from typing import Dict, List, Optional
from queue import Queue

from utils import addCardToDict, getCardDict, findAllCombs, filterActions, updateCardDictAfterAction, groupActions,\
    getChoiceUnderSameAction, getChoiceUnderThreeWithTwo, canPlayAllInOnce, canPassOnly, passIsFine
from guandan_tree_node import GDNode
from cardcomb import CardComb

class ProgressSimulator(object):
    
    def __init__(self):
        pass

    @staticmethod
    def simulate_two_players(cd1 : Dict[str, int], cd2 : Dict[str, int], level : int) -> GDNode:
        node_queue = Queue()
        
        top_node = GDNode(0, 0, cd1.copy(), cd2.copy(), 0)
        
        if cd1['total'] == 0:
            top_node.reward = 1
            return top_node
        elif cd2['total'] == 0:
            top_node.reward = -1
            return top_node
        
        node_queue.put(top_node)
        n_index = 1
        
        while not node_queue.empty():
            node : GDNode = node_queue.get_nowait()
            print(f"node num = {node.node_index}")
            
            if node.player_index == 0 and node.card_dict2['total'] == 0:
                node.reward = -1
                continue
            elif node.player_index == 1 and node.card_dict1['total'] == 0:
                node.reward = 1
                continue
            
            comb1 = findAllCombs(node.card_dict1, level)
            comb2 = findAllCombs(node.card_dict2, level)
            base_action = node.current_greatest_action
            
            acts1 = filterActions(comb1, base_action, level)
            acts2 = filterActions(comb2, base_action, level)
            
            if node.player_index == 0:
                if len(acts1) == 1 and acts1[0].is_pass():
                    next_node : GDNode = GDNode(1, n_index, node.card_dict1.copy(), node.card_dict2.copy(), node.layer_num + 1)
                    node.add_child_node(next_node, CardComb.pass_cardcomb())
                    node_queue.put(next_node)
                    n_index += 1
                else:
                    idx1 = canPlayAllInOnce(acts1, node.card_dict1['total'])
                    if idx1 >= 0:
                        new_cd1 = updateCardDictAfterAction(node.card_dict1, acts1[idx1])
                        next_node : GDNode = GDNode(1, n_index, new_cd1, node.card_dict2.copy(), node.layer_num + 1, acts1[idx1])
                        node.add_child_node(next_node, acts1[idx1])
                        node_queue.put(next_node)
                        n_index += 1
                    else:
                        group_actions = groupActions(acts1)
                        for group in group_actions:
                            if len(group[1]) == 1:
                                if group[0] == 0 and passIsFine(comb2, node.card_dict2['total']):
                                    next_node : GDNode = GDNode(1, n_index, node.card_dict1.copy(), node.card_dict2.copy(), node.layer_num + 1, CardComb.pass_cardcomb())
                                    node.add_child_node(next_node, CardComb.pass_cardcomb())
                                    node_queue.put(next_node)
                                    n_index += 1
                                else:
                                    new_cd1 = updateCardDictAfterAction(node.card_dict1, group[1][0])
                                    next_node : GDNode = GDNode(1, n_index, new_cd1, node.card_dict2.copy(), node.layer_num + 1, group[1][0])
                                    node.add_child_node(next_node, group[1][0])
                                    node_queue.put(next_node)
                                    n_index += 1
                            else:
                                these_actions = group[1]
                                choice : Optional[CardComb] = None
                                # ThreeWithTwo
                                if group[0] >= 44 and group[0] <= 56:
                                    choice = getChoiceUnderThreeWithTwo(node.card_dict1, comb1, these_actions, level)
                                # Other
                                else:
                                    choice = getChoiceUnderSameAction(node.card_dict1, these_actions, level)
                                new_cd1 = updateCardDictAfterAction(node.card_dict1, choice)
                                next_node : GDNode = GDNode(1, n_index, new_cd1, node.card_dict2.copy(), node.layer_num + 1, choice)
                                node.add_child_node(next_node, acts1[idx1])
                                node_queue.put(next_node)
                                n_index += 1
                        
            else:
                if len(acts2) == 1 and acts2[0].is_pass():
                    next_node : GDNode = GDNode(0, n_index, node.card_dict1.copy(), node.card_dict2.copy(), node.layer_num + 1)
                    node.add_child_node(next_node, CardComb.pass_cardcomb())
                    node_queue.put(next_node)
                    n_index += 1
                else:
                    idx2 = canPlayAllInOnce(acts2, node.card_dict2['total'])
                    if idx2 >= 0:
                        new_cd2 = updateCardDictAfterAction(node.card_dict2, acts2[idx2])
                        next_node : GDNode = GDNode(0, n_index, node.card_dict1.copy(), new_cd2, node.layer_num + 1, acts2[idx2])
                        node.add_child_node(next_node, acts2[idx2])
                        node_queue.put(next_node)
                        n_index += 1
                    else:
                        group_actions = groupActions(acts2)
                        for group in group_actions:
                            if len(group[1]) == 1:
                                if group[0] == 0 and passIsFine(comb1, node.card_dict1['total']):
                                    next_node : GDNode = GDNode(0, n_index, node.card_dict1.copy(), node.card_dict2.copy(), node.layer_num + 1, CardComb.pass_cardcomb())
                                    node.add_child_node(next_node, CardComb.pass_cardcomb())
                                    node_queue.put(next_node)
                                    n_index += 1
                                else:
                                    new_cd2 = updateCardDictAfterAction(node.card_dict2, group[1][0])
                                    next_node : GDNode = GDNode(0, n_index, node.card_dict1.copy(), new_cd2, node.layer_num + 1, group[1][0])
                                    node.add_child_node(next_node, group[1][0])
                                    node_queue.put(next_node)
                                    n_index += 1
                            else:
                                these_actions = group[1]
                                choice : Optional[CardComb] = None
                                # ThreeWithTwo
                                if group[0] >= 44 and group[0] <= 56:
                                    choice = getChoiceUnderThreeWithTwo(node.card_dict2, comb2, these_actions, level)
                                # Other
                                else:
                                    choice = getChoiceUnderSameAction(node.card_dict2, these_actions, level)
                                new_cd2 = updateCardDictAfterAction(node.card_dict2, choice)
                                next_node : GDNode = GDNode(0, n_index, node.card_dict1, new_cd2, node.layer_num + 1, choice)
                                node.add_child_node(next_node, choice)
                                node_queue.put(next_node)
                                n_index += 1

        _ = top_node.update_recursively()

        return top_node


if __name__ == "__main__":
    d1 = getCardDict()
    d2 = getCardDict()
    
    c1 = ['H2', 'H3', 'D4', 'D6', 'D9']
    c2 = ['C2', 'C4', 'S5', 'D9', 'SJ']
    # c1 = ['H5', 'D9']
    # c2 = ['C2', 'CK']
    
    addCardToDict(d1, c1)
    addCardToDict(d2, c2)
    
    level = 13
    
    node = ProgressSimulator.simulate_two_players(d1, d2, level)
    
    print(node.reward)
        