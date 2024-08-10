from typing import List, Dict, Tuple, Optional, Union
from queue import Queue

import tensorflow as tf
import itertools as it
import secrets
import copy

from utils import NumToCardNum, getWildCard, randomCardDict, getCardDict, findAllPairs, findAllCombs, filterActions, \
    updateCardDictAfterAction, passIsFine, addCardToDict, canPlayAllInOnce, canPassOnly

from guandan_tree_node import GDNode, GDResultNode

def cardDictToListWithMode(card_dict : Dict[str, int], mode : int, level : int) -> tf.Tensor:
    """
    @param mode:
    0 : include all cards
    1 : include all cards except SB and HR
    2 : include all cards with suit, from A to K then A.
    """
    if mode == 0:
        ranking = [i for i in range(1, 14)]
        if level != 13:
            temp1 = ranking[0:level - 1]
            temp2 = ranking[level:]
            ranking = temp1 + temp2 + [level]
        str_ranking = []
        for r in ranking:
            if r < 9:
                str_ranking.append(str(r + 1))
            else:
                str_ranking.append(NumToCardNum[r + 1])
        answer = [0] * 15
        for i in range(13):
            answer[i] += card_dict[f"S{str_ranking[i]}"]
            answer[i] += card_dict[f"H{str_ranking[i]}"]
            answer[i] += card_dict[f"C{str_ranking[i]}"]
            answer[i] += card_dict[f"D{str_ranking[i]}"]
        answer[13] = card_dict['SB']
        answer[14] = card_dict['HR']
        return tf.reshape(answer, [15, 1])
    elif mode == 1:
        ranking = [i for i in range(1, 14)]
        if level != 13:
            temp1 = ranking[0:level - 1]
            temp2 = ranking[level:]
            ranking = temp1 + temp2 + [level]
        str_ranking = []
        for r in ranking:
            if r < 9:
                str_ranking.append(str(r + 1))
            else:
                str_ranking.append(NumToCardNum[r + 1])
        answer = [0] * 15
        for i in range(13):
            answer[i] += card_dict[f"S{str_ranking[i]}"]
            answer[i] += card_dict[f"H{str_ranking[i]}"]
            answer[i] += card_dict[f"C{str_ranking[i]}"]
            answer[i] += card_dict[f"D{str_ranking[i]}"]
        return tf.reshape(answer, [13, 1])
    elif mode == 2:
        str_ranking = []
        for i in range(1, 15):
            if i != 1 and i < 10:
                str_ranking.append(str(i))
            else:
                str_ranking.append(NumToCardNum[i])
        answer = []
        for i in range(14):
            temp = []
            temp.append(card_dict[f"S{str_ranking[i]}"])
            temp.append(card_dict[f"H{str_ranking[i]}"])
            temp.append(card_dict[f"C{str_ranking[i]}"])
            temp.append(card_dict[f"D{str_ranking[i]}"])
            answer.append(temp)
        return tf.reshape(answer, shape=[14, 4])
            
                
    else:
        raise ValueError("@param mode must be either 0, 1 or 2.")

def findAll_2_2_comb(level : int = 13) -> List[Tuple[str, str]]:
    wild_card = getWildCard(level)
    card_dict = randomCardDict(108)
    temp = findAllPairs(card_dict, wild_card)
    all_pairs : List[Tuple[str, str]] = list()
    
    for pair in temp:
        new_pair = []
        if pair[0] == wild_card:
            new_pair.append("*")
        else:
            new_pair.append(pair[0])
        if pair[1] == wild_card:
            new_pair.append("*")
        else:
            new_pair.append(pair[1])
        all_pairs.append(tuple(new_pair))
    
    all_situations = list(it.product(all_pairs, all_pairs.copy()))
    
    final_situation = list()

    for situation in all_situations:
        # situation = ((...,...),(...,...))
        legal = True
        counter = dict()
        temp = situation[0] + situation[1]
        for card in temp:
            if card in counter:
                if counter[card] == 2:
                    legal = False
                    break
                else:
                    counter[card] += 1
            else:
                counter[card] = 1
        if legal:
            final_situation.append(situation)
    
    print(len(final_situation))
    
    return final_situation

def simulate_two_card_dicts(cd1 : Dict[str, int], cd2 : Dict[str, int], level : int) -> Tuple[GDNode, List[GDNode]]:
    '''
    This function returns the result of player1 use @param cd1 to play against player2 with @param cd2.\n
    @output the top node and all leaf nodes, which is sorted by their layers in ascending order.\n
    If player1 wins player2 in some way, the reward of top node is 1; Else, the reward of top node is -1 \
        (player1 cannot win player2 even if player1 plays first).
    '''
    node_queue = Queue()
    leaf_nodes = list()
    
    top_node = GDNode(0, 0, cd1.copy(), cd2.copy(), 0, level, None)
    
    if cd2['total'] == 0:
        leaf_nodes.append(top_node)
        leaf_nodes.append(None)
        return top_node, leaf_nodes
    
    node_queue.put(top_node)
    n_index = 1
    
    while not node_queue.empty():
        node : GDNode = node_queue.get_nowait() 
        index = node.player_index
        l = node.layer_num
        all_actions = findAllCombs(node.card_dict1, level)
        legal_actions = filterActions(all_actions, node.current_greatest_action, level)
        if len(legal_actions) == 1 and legal_actions[0].is_pass():
            next_node = GDNode((index + 1) % 2, n_index, node.card_dict2.copy(), node.card_dict1.copy(), l + 1, level, None)
            n_index += 1
            node.add_child_node(next_node, None)
            next_node.add_parent(node)
            node_queue.put(next_node)
            continue
        play_all_in_once : int = canPlayAllInOnce(legal_actions, node.card_dict1['total'])
        if play_all_in_once > -1:
            perfect_action = legal_actions[play_all_in_once]
            next_node = GDNode((index + 1) % 2, n_index, node.card_dict2.copy(), getCardDict(), l + 1, level, perfect_action)
            if index == 0:
                next_node.set_reward(1)
            else:
                next_node.set_reward(-1)
            node.add_child_node(next_node, perfect_action)
            next_node.add_parent(node)
            n_index += 1
            leaf_nodes.append([next_node, node])
        else:
            index_of_action : int = -1
            left_dict : Optional[Dict[str, int]] = None
            oppo_actions = findAllCombs(node.card_dict2, level)
            for i in range(len(legal_actions)):
                action = legal_actions[i]
                if not action.is_pass():
                    left_dict = updateCardDictAfterAction(node.card_dict1, action)
                    left_actions = findAllCombs(left_dict, level)
                    a = canPassOnly(action, oppo_actions, level)
                    if a and canPlayAllInOnce(left_actions, left_dict['total']) > -1:
                        index_of_action = i
                        break
            if index_of_action > -1:
                next_node = GDNode((index + 1) % 2, n_index, node.card_dict2.copy(), left_dict.copy(), l + 1, level, legal_actions[index_of_action])
                n_index += 1
                node.add_child_node(next_node, legal_actions[index_of_action])
                next_node.add_parent(node)
                node_queue.put(next_node)
            else:
                can_pass = True
                if node.current_greatest_action is not None and not passIsFine(oppo_actions, node.card_dict2['total']):
                    can_pass = False
                temp = list()
                temp2 = list()
                for a in legal_actions:
                    if a.is_pass() and can_pass:
                        next_node = GDNode((index + 1) % 2, n_index, node.card_dict2.copy(), node.card_dict1.copy(), l + 1, level, None)
                        n_index += 1
                        temp.append((next_node, None, False))
                    elif not a.is_pass():
                        left_dict = updateCardDictAfterAction(node.card_dict1, a)
                        next_node = GDNode((index + 1) % 2, n_index, node.card_dict2.copy(), left_dict, l + 1, level, a)
                        n_index += 1
                        if left_dict['total'] == 0:
                            temp2.append(next_node)
                            temp.append((next_node, a, True))
                            if node.player_index == 0:
                                next_node.set_reward(1)
                            else:
                                next_node.set_reward(-1)
                        else:
                            temp.append((next_node, a, False))
                for t in temp:
                    node.add_child_node(t[0], t[1])
                    t[0].add_parent(node)
                    if not t[2]:
                        node_queue.put(t[0])
                
                if len(temp2) > 0:
                    temp2.append(node)
                    leaf_nodes.append(temp2)

    return top_node, leaf_nodes

def build_result_tree(top_node : GDNode) -> Tuple[GDResultNode, List[List[GDResultNode]]]:
    if top_node.children is None:
        node = GDResultNode(top_node.player_index, 0, 0, top_node.reward)
        n_list = [node]
        return node, n_list
    
    queue = Queue()
    queue2 = Queue()
    n_index = 0
    
    answer = GDResultNode(top_node.player_index, n_index, 0, top_node.reward)
    
    n_index = 1
    
    queue.put(top_node)
    queue2.put(answer)
    
    leaf_nodes = list()
    
    while not queue.empty():
        current_node : GDNode = queue.get_nowait()
        last_result_node : GDResultNode = queue2.get_nowait()
        temp = list()
        if current_node.children is not None:
            for child in current_node.children:
                new_node = GDResultNode(child.player_index, n_index, child.layer_num, child.reward)
                if child.children is None or len(child.children) == 0:
                    temp.append(new_node)
                last_result_node.add_child_node(new_node, list())
                new_node.add_parent(last_result_node)
                queue.put(child)
                queue2.put(new_node)
                n_index += 1
        if len(temp) > 0:
            temp.append(last_result_node)
            leaf_nodes.append(temp)
    
    return answer, leaf_nodes    

def update_search_tree(leaf_nodes : List[List[Optional[Union[GDResultNode, GDNode]]]]) -> Union[GDResultNode, GDNode]:
    '''
    This function will change @param leaf_nodes.\n
    Please do not use it after calling this function.\n
    '''
    layer = leaf_nodes[-1][0].layer_num
    top_node : Optional[Union[GDResultNode, GDNode]] = None
    while layer >= 0:
        temp : List[Union[GDResultNode, GDNode]] = list()
        while len(leaf_nodes) > 0 and leaf_nodes[-1][0].layer_num == layer:
            node_group = leaf_nodes.pop()
            parent : Optional[Union[GDResultNode, GDNode]] = node_group[-1]
            if parent is None:
                top_node = node_group[0]
                break
            player_index = node_group[0].player_index
            can_win : bool = False if player_index == 1 else True
            if player_index == 1:
                for node in node_group:
                    if node.reward == 1:
                        can_win = True
                        break
            else:
                for node in node_group:
                    if node.reward == -1:
                        can_win = False
                        break
            if can_win:
                parent.set_reward(1)
                # if isinstance(node_group[0], GDNode):
                #     for i in range(len(node_group) - 1):
                #         if node_group[i].reward == 1:
                #             if parent.best_child_index is None:
                #                 parent.best_child_index = list()
                #             parent.best_child_index.append(i)
            else:
                parent.set_reward(-1)
            assert parent.reward != -100
            temp.append(parent)
        if top_node is None:
            for p in temp:
                index : int = -1
                for index in range(len(leaf_nodes) - 1, -1, -1):
                    l2 = leaf_nodes[index][0].layer_num
                    if l2 < layer - 1:
                        index = -1
                        break
                    p2 = leaf_nodes[index][-1]
                    if p2 in p.parent:
                        break
                if index == -1:
                    if p.parent is None:
                        next_group = [p, None]
                    else:
                        next_group = [p, list(p.parent)[0]]
                    leaf_nodes.append(next_group)
                else:
                    leaf_nodes[index].insert(0, p)
        layer -= 1

    return top_node

def get_progress_from_node(top_node : GDNode) -> List[List[GDNode]]:
    '''
    Currently, we assume the reward for the node is 1.
    '''
    progress = Queue()
    queue = Queue()
    final_progress : List[List[GDNode]] = list()
    
    progress.put([top_node])
    queue.put(top_node)
    
    while not queue.empty():
        current_node : GDNode = queue.get_nowait()
        current_progress : List[GDNode] = progress.get_nowait()
        if current_node.player_index == 0:
            next_node : Optional[GDNode] = None
            for node in current_node.children:
                if node.reward == 1:
                    next_node = node
                    break
            if next_node is None:
                raise ValueError("This node has no child with positive reward!")
            current_progress.append(next_node)
            if not next_node.is_leaf():
                queue.put(next_node)
                progress.put(current_progress)
            else:
                final_progress.append(current_progress)
        else:
            temp : List[GDNode] = list()
            for node in current_node.children:
                assert next_node.reward == 1
                temp.append(node)
            t = 0
            temp2 = current_progress.copy()
            for node in temp:
                if t == 0:
                    current_progress.append(node)
                    if not node.is_leaf():
                        progress.put(current_progress)
                        queue.put(node)
                else:
                    temp3 = temp2.copy()
                    temp3.append(node)
                    if not node.is_leaf():
                        progress.put(temp3)
                        queue.put(node)
                t = 1
    
    return final_progress

def get_action_from_progress_nodes(progresses : List[List[GDNode]]) -> List[List[Tuple]]:
    answer = list()
    
    for progress in progresses:
        temp = list()
        for i in range(len(progress) - 1):
            node = progress[i]
            action = node.actions[node.children.index(progress[i + 1])]
            t : Tuple = (node.player_index, action)
            temp.append(t)
        answer.append(temp)
    
    return answer

def build_dummy_result_tree() -> Tuple[GDResultNode, List[List[GDResultNode]]]:
    top_node = GDResultNode(0, 0, 0, -100)
    node1 = GDResultNode(1, 1, 1, 1)
    node2 = GDResultNode(1, 2, 1, 1)
    node3 = GDResultNode(1, 3, 1, 1)
    top_node.add_child_node(node1, None)
    top_node.add_child_node(node2, None)
    top_node.add_child_node(node3, None)
    return top_node, [[node1, node2, node3, top_node]]

def print_node_group(node_group : List[Optional[GDResultNode]]) -> None:
    parent : Optional[GDResultNode] = node_group[-1]
    t : str = "["
    if parent is None:
        for node in node_group:
            if node is not None:
                t += f"node {node.node_index} - {node.player_index}"
            if node != parent:
                t += ", "
            else:
                t += "]"
    else:
        for node in node_group:
            t += f"node {node.node_index} - {node.player_index}"
            if node != parent:
                t += ", "
            else:
                t += "]"
    print(t)

if __name__ == "__main__":
    d1 = getCardDict()
    d2 = getCardDict()
    
    level = 13
    
    # hand card1 = [H3, H7]
    _ = addCardToDict(['S5', 'S7', 'H4'], d2)

    # hand card2 = [H5, D5]
    _ = addCardToDict(['S7', 'S9', 'HT'], d1)
    
    top_node, leaf_nodes = simulate_two_card_dicts(d1, d2, level)
    
    # dummy_top_node, leaf_nodes = build_dummy_result_tree()
    
    # top_node2 = update_search_tree(leaf_nodes)
    
    # print(top_node2.node_index)
    
    # result_top_node, leaf_nodes = build_result_tree(top_node)
    
    # top_node2 = update_search_tree(leaf_nodes)
    
    r = top_node.update_recursively()
    
    if r == 1:
        progress = get_progress_from_node(top_node)
        final_progress = get_action_from_progress_nodes(progress)
        for p in final_progress:
            print("==========")
            print(p)
            print("*********")
    else:
        print("Player0 cannot win Player1")