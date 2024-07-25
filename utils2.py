from typing import List, Dict, Tuple, Set, Optional
from queue import Queue

import tensorflow as tf
import itertools as it

from utils import POWERS, SUITS, NumToCardNum, getHeartLevelCard, randomCardDict, getCardDict, findAllPairs, findAllCombs, filterActions, \
    updateCardDictAfterAction, passIsFine, addCardToDict

from guandan_tree_node import GDNode

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

def convertDictByLevel(card_dict : Dict[str, int], level : int) -> Dict[str, int]:
    p = POWERS.copy()
    _ = p.pop(level - 1)
    if level >= 9:
        p.insert(12, NumToCardNum[level + 1])
    else:
        p.insert(12, str(level + 1))
    p2 = POWERS.copy()
    final_dict = getCardDict()
    origin_wild_card = getHeartLevelCard(level)
    
    del final_dict['HA']
    final_dict['*'] = 0
    
    if level == 13:
        keys = card_dict.keys()
        for key in keys:
            if key == 'HA':
                final_dict['*'] = card_dict[key]
            else:
                final_dict[key] = card_dict[key]
    else:
        for i in range(13):
            for j in range(4):
                origin = SUITS[j] + p[i]
                if origin == origin_wild_card:
                    final_dict['*'] = card_dict[origin]
                else:
                    new = SUITS[j] + p2[i]
                    final_dict[new] = card_dict[origin]
    
    return final_dict           

def findAll_1_1_comb(level : int = 13) -> List[Tuple[str, str]]:
    current_comb : Set[Tuple[str, str]] = set()

def findAll_2_2_comb(level : int = 13) -> List[Tuple[str, str]]:
    wild_card = getHeartLevelCard(level)
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

def simulate_two_card_dicts(cd1 : Dict[str, int], cd2 : Dict[str, int], level : int) -> GDNode:
    '''
    return the result of player1 use @param cd1 to against player2 with @param cd2.\n
    @output is either 1 or -1.\n
    If player1 wins player2 in some way, return 1; Else, return -1 (player1 cannot win player2 even if player1 play first).
    '''
    node_queue = Queue()
    leaf_nodes = list()
    all_actions = findAllCombs(cd1, level)
    
    print(all_actions)
    
    top_node = GDNode(0, cd1.copy(), cd2.copy(), 0, level, None)
    
    for action in all_actions:
        left_dict = updateCardDictAfterAction(cd1, action)
        node = GDNode(1, cd2, left_dict, 1, level, action)
        if left_dict['total'] == 0:
            leaf_nodes.append(node)
        node.add_parent(top_node)
        top_node.add_child_node(node, action)
        if left_dict['total'] > 0:
            node_queue.put(node)
    
    while not node_queue.empty():
        node : GDNode = node_queue.get_nowait() 
        index = node.player_index
        l = node.layer_num
        all_actions = findAllCombs(node.card_dict1, level)
        legal_actions = filterActions(all_actions, node.current_greatest_action, level)
        can_pass = True
        oppo_actions = findAllCombs(node.card_dict2, level)
        if len(legal_actions) == 1 and legal_actions[0][0] == 'PASS':
            can_pass = True
        elif node.current_greatest_action is not None and not passIsFine(oppo_actions, node.card_dict2['total']):
            can_pass = False
        temp = list()
        for a in legal_actions:
            if a[0] == 'PASS' and can_pass:
                next_node = GDNode((index + 1) % 2, node.card_dict2.copy(), node.card_dict1.copy(), l + 1, level, None)
                temp.append((next_node, None, False))
            elif a[0] != 'PASS':
                left_dict = updateCardDictAfterAction(node.card_dict1, a)
                next_node = GDNode((index + 1) % 2, node.card_dict2.copy(), left_dict, l + 1, level, a.copy())
                if left_dict['total'] == 0:
                    leaf_nodes.append(next_node)
                    temp.append((next_node, a, True))
                else:
                    temp.append((next_node, a, False))
        for t in temp:
            node.add_child_node(t[0], t[1])
            t[0].add_parent(node)
            if not t[2]:
                node_queue.put(t[0])
    
    for node in leaf_nodes:
        if node.player_index == 1:
            node.set_reward(1)
        else:
            node.set_reward(-1)
    
    return top_node

def get_progress_from_node(node : GDNode) -> Tuple[str, List]:
    pass    

if __name__ == "__main__":
    d1 = getCardDict()
    d2 = getCardDict()
    
    # hand card1 = [H3, H7]
    _ = addCardToDict(['H3', 'H3', 'S3', 'HK'], d1)
    # hand card2 = [H5, D5]
    _ = addCardToDict(['H7', 'SA', 'HA'], d2)
    
    level = 13
    
    top_node = simulate_two_card_dicts(d1, d2, level)