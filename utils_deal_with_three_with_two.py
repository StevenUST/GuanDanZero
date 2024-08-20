from typing import Dict, List
from copy import deepcopy
from utils import findAllPairs, findAllTrips, findAllBombs, findAllStraightFlushes

def find_three_same_elements(lst):
    for element in lst:
        if lst.count(element) == 3:
            return element
    return None


def count_total_value(cards: Dict[str, int], target_value: str) -> int:
    total_count = 0
    for card, count in cards.items():
        if card[1:] == target_value:
            total_count += count
    return total_count


def merge_unique_elements(nested_list: list[list[str]]) -> List[str]:
    unique_elements = set()
    for sublist in nested_list:
        unique_elements.update(sublist)
    return list(unique_elements)


def clear_specific_card(cards: Dict[str, int], target_value: str) -> Dict[str, int]:
    new_cards = deepcopy(cards)  # 复制原始字典
    for card in new_cards:  # 遍历字典的所有键
        if card[1:] == target_value:  # 检查牌的数值是否等于目标值
            new_cards[card] = 0  # 将符合条件的牌的数量设为0
    return new_cards


def remove_pairs_with_card_in_list(pairs_list: List[List[str]], listB: List[str]) -> List[List[str]]:
    # 创建一个新列表来保存符合条件的对子
    filtered_pairs = []

    for pair in pairs_list:
        # 检查对子中的任意一张牌是否在 listB 中
        if not any(card in listB for card in pair):
            filtered_pairs.append(pair)  # 如果对子中的牌都不在 listB 中，则保留该对子

    return filtered_pairs


def create_card_rankings(special_card: str):
    # 初始的固定牌值顺序，2 最小，大小王最大
    base_ranking = {
        '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
        '7': 6, '8': 7, '9': 8, '10': 9, 'J': 10,
        'Q': 11, 'K': 12, 'A': 13,
        '小王': 14, '大王': 15
    }

    # 动态调整特殊牌的排名，仅次于大小王
    if special_card in base_ranking:
        base_ranking[special_card] = 13.5  # 特殊牌排名仅次于 A (13) 并且小于小王 (14)

    return base_ranking


def sort_pairs_by_rank(pairs_list: list[list[str]], rank) -> list[list[str]]:
    # 定义一个辅助函数，获取对子中最小的牌的排名
    def get_pair_rank(pair: list[str]) -> int:
        return min(rank[pair[0][1:]], rank[pair[1][1:]])  # 根据对子中的牌获取其排名

    # 使用排序函数对对子进行排序
    sorted_pairs = sorted(pairs_list, key=get_pair_rank)

    return sorted_pairs


def find_suits_with_card(cards, target_value: str) -> List[str]:
    suits_with_card = []
    for card in cards.keys():
        if card[1:] == target_value:  # 检查牌面值是否为目标值
            suits_with_card.append(card)  # 将符合条件的牌加入列表中
    return suits_with_card


def count_specific_cards(cards: Dict[str, int], target_value: str, excluded_cards: List[str]) -> int:
    count = 0
    for card in cards.keys():
        if card[1:] == target_value and card not in excluded_cards:
            count += cards[card]  # 累加符合条件的牌的数量
    return count


def remove_k_cards1(cards: Dict[str, int], excluded_cards: list[str], target_value: str, remove_count: int) -> Dict[str, int]:
    updated_cards = deepcopy(cards)

    # 从集合['HK']以外的其他花色的K中删除指定数量的牌
    for card in updated_cards.keys():
        # 检查花色和牌面值是否符合条件
        if card not in excluded_cards and card[1:] == target_value:
            if updated_cards[card] >= remove_count:
                updated_cards[card] -= remove_count  # 直接减去指定数量的牌
                break  # 如果满足要求的牌已处理完毕，退出循环
            else:
                # 如果当前花色的K数量不足，减去它并继续处理其他花色
                remove_count -= updated_cards[card]
                updated_cards[card] = 0  # 当前花色的K全部移除

    return updated_cards


def remove_k_cards2(cards: Dict[str, int], excluded_cards: list[str], target_value: str, remove_count: int) -> Dict[str, int]:
    # 复制原来的手牌字典
    updated_cards = deepcopy(cards)

    # 先从非 excluded_cards 中移除指定数量的 K 牌
    for card in updated_cards.keys():
        # 检查是否为非排除牌且是 K
        if card not in excluded_cards and card[1:] == target_value:
            if updated_cards[card] >= remove_count:
                updated_cards[card] -= remove_count
                return updated_cards  # 如果足够移除，直接返回结果
            else:
                remove_count -= updated_cards[card]  # 如果不足，移除所有并减少计数
                updated_cards[card] = 0

    # 如果非 excluded_cards 中不足，则从 excluded_cards 中移除剩余数量的 K 牌
    for card in excluded_cards:
        if card in updated_cards and card[1:] == target_value:  # 确保卡片存在并且是 K
            if updated_cards[card] >= remove_count:
                updated_cards[card] -= remove_count
                return updated_cards  # 移除剩余的数量后返回结果
            else:
                remove_count -= updated_cards[card]  # 如果不足，移除所有并减少计数
                updated_cards[card] = 0

    return updated_cards


def remove_pairs_with_triple(pairs_list: List[List[str]], Triple) -> List[List[str]]:
    # 过滤掉包含K的对子
    filtered_pairs = [
        pair for pair in pairs_list if Triple not in pair[0] and Triple not in pair[1]]
    return filtered_pairs


def pair_for_Triplewithtwo(player_card: Dict[str, int], wild_card : str, triplewithtwo: List[str]):
    Triple = find_three_same_elements(triplewithtwo)
    all_num_for_Triple = count_total_value(player_card, Triple)
    if all_num_for_Triple == 3:
        new_card = clear_specific_card(player_card, Triple)

        all_pair = findAllPairs(new_card, wild_card)
        all_bomb = findAllBombs(new_card, wild_card)
        element_for_all_bomb = merge_unique_elements(all_bomb)
        all_straightflush = findAllStraightFlushes(new_card, wild_card)
        element_for_all_straightflush = merge_unique_elements(
            all_straightflush)
        union = set(element_for_all_bomb) | set(element_for_all_straightflush)
        all_pair = remove_pairs_with_card_in_list(all_pair, list(union))
        if len(all_pair) == 0:
            return None
        else:
            rank = create_card_rankings(wild_card[1])
            final_pairs = sort_pairs_by_rank(all_pair, rank)
            return final_pairs[0]

    else:
        all_straightflush = findAllStraightFlushes(player_card, wild_card)
        if len(all_straightflush) != 0:
            element_for_all_straightflush = merge_unique_elements(
                all_straightflush)
            all_type = find_suits_with_card(player_card, Triple)
            imp = list(set(element_for_all_straightflush) & set(all_type))
            num = count_specific_cards(player_card, Triple, imp)
            if num >= 3:
                new_card = remove_k_cards1(player_card, imp, Triple, 3)

                all_pair = findAllPairs(new_card, wild_card)
                all_bomb = findAllBombs(new_card, wild_card)
                element_for_all_bomb = merge_unique_elements(all_bomb)
                all_straightflush = findAllStraightFlushes(new_card, wild_card)
                element_for_all_straightflush = merge_unique_elements(
                    all_straightflush)
                union = set(element_for_all_bomb) | set(
                    element_for_all_straightflush)
                all_pair = remove_pairs_with_card_in_list(
                    all_pair, list(union))
                if len(all_pair) == 0:
                    return None
                else:
                    rank = create_card_rankings(wild_card[1])
                    final_pairs = sort_pairs_by_rank(all_pair, rank)
                    return final_pairs[0]
            else:
                new_card = remove_k_cards2(player_card, imp, Triple, 3)

                all_pair = findAllPairs(new_card, wild_card)
                all_bomb = findAllBombs(new_card, wild_card)
                element_for_all_bomb = merge_unique_elements(all_bomb)
                all_straightflush = findAllStraightFlushes(new_card, wild_card)
                element_for_all_straightflush = merge_unique_elements(
                    all_straightflush)
                union = set(element_for_all_bomb) | set(
                    element_for_all_straightflush)
                all_pair = remove_pairs_with_card_in_list(
                    all_pair, list(union))
                if len(all_pair) == 0:
                    return None
                else:
                    rank = create_card_rankings(wild_card[1])
                    final_pairs = sort_pairs_by_rank(all_pair, rank)
                    return final_pairs[0]
        else:
            all_pair = findAllPairs(player_card, wild_card)
            all_pair = remove_pairs_with_triple(all_pair, Triple)
            rank = create_card_rankings(wild_card[1])
            final_pairs = sort_pairs_by_rank(all_pair, rank)
            return final_pairs[0]