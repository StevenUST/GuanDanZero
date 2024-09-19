from typing import Tuple, Final, Dict, List

from cardcomb import CardCombNoSuit
from utils import getWildCard

NumToCardNum: Final[Dict[int, str]] = {
    0: 'PASS', 1: 'A', 2: '2', 3: '3', 4 : '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: 'B', 16: 'R'
}

CardNumToNum: Final[Dict[str, int]] = {
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, 'B': 15, 'R': 16
}

def getCardDictWithoutSuit() -> Dict[int, int]:
    '''
    d[1] : Number of card with rank 2\n
    [Continue]\n
    d[9] : Number of card with rank T\n
    d[13] : Number of card with rank A\n
    d[16] : Number of wild card\n
    d[17] : Total number of cards in the Dict
    '''
    d = dict()
    for i in range(1, 17):
        d[i] = 0
    d[17] = 0
    return d

def findAllCombWithoutSuit(cards : List[str], card_dict : Dict[str, int], level : int) -> List[CardCombNoSuit]:
    answer = list()
    if card_dict[17] == 0:
        return answer
    
    answer.extend(findAllSingleWithoutSuit(card_dict, level))
    answer.extend(findAllPairWithoutSuit(card_dict, level))
    answer.extend(findAllTripWithouSuit(card_dict))
    answer.extend(findAllThreeWithTwoWithouSuit(card_dict))
    answer.extend(findAllBomb(card_dict))
    
    straight_data = isStraight(cards, level)
    if straight_data[0] > -1:
        if straight_data[1]:
            answer.append(CardCombNoSuit('StraightFlush', straight_data[0], 0, 5, card_dict[16]))
        else:
            answer.append(CardCombNoSuit('Straight', straight_data[0], 0, 5, card_dict[16]))
    
    return answer
    
def findAllSingleWithoutSuit(card_dict : Dict[str, int], level : int) -> List[CardCombNoSuit]:
    answer = list()
    
    for i in range(1, 15):
        if card_dict[i] > 0:
            answer.append(CardCombNoSuit('Single', i, 0, 1, 0))
    
    if card_dict[16] > 0:
        answer.append(CardCombNoSuit('Single', level, 0, 1, 1))
    
    return answer

def findAllPairWithoutSuit(card_dict : Dict[str, int], level : int) -> List[CardCombNoSuit]:
    answer = list()
    
    if card_dict[17] < 2:
        return answer
    
    for i in range(1, 16):
        if card_dict[i] > 1:
            answer.append(CardCombNoSuit('Pair', i, 0, 2, 0))
        elif card_dict[i] == 1 and i < 14 and card_dict[16] >= 1:
            answer.append(CardCombNoSuit('Pair', i, 0, 2, 1))
    
    if card_dict[16] == 2:
        answer.append(CardCombNoSuit('Pair', level, 0, 2, 2))
    
    return answer

def findAllTripWithouSuit(card_dict : Dict[str, int]) -> List[CardCombNoSuit]:
    answer = list()
    
    if card_dict[17] < 3:
        return answer
    
    for i in range(1, 14):
        if card_dict[i] > 2:
            answer.append(CardCombNoSuit('Trip', i, 0, 3, 0))
        elif card_dict[i] == 2 and card_dict[16] >= 1:
            answer.append(CardCombNoSuit('Trip', i, 0, 3, 1))
        elif card_dict[i] == 1 and card_dict[16] == 2:
            answer.append(CardCombNoSuit('Trip', i, 0, 3, 2))

def findAllThreeWithTwoWithouSuit(card_dict : Dict[str, int]) -> List[CardCombNoSuit]:
    answer = list()
    
    if card_dict[17] < 5:
        return answer
    
    wild_card_num = card_dict[16]
    
    for i in range(1, 14):
        temp_num = wild_card_num
        if card_dict[i] == 3:
            for j in list(filter(lambda val : val != i, list(range(1, 16)))):
                if card_dict[j] == 2:
                    answer.append(CardCombNoSuit('ThreeWithTwo', i, j, 5, 0))
                elif card_dict[j] == 1 and j < 14 and temp_num >= 1:
                    answer.append(CardCombNoSuit('ThreeWithTwo', i, j, 5, 1))
        elif card_dict[i] == 2 and temp_num >= 1:
            temp_num -= 1
            for j in list(filter(lambda val : val != i, list(range(1, 16)))):
                if card_dict[j] == 2:
                    answer.append(CardCombNoSuit('ThreeWithTwo', i, j, 5, 1))
                elif card_dict[j] == 1 and j < 14 and temp_num >= 1:
                    answer.append(CardCombNoSuit('ThreeWithTwo', i, j, 5, 2))
        elif card_dict[i] == 1 and temp_num == 2:
            for j in list(filter(lambda val : val != i, list(range(1, 16)))):
                if card_dict[j] == 2:
                    answer.append(CardCombNoSuit('ThreeWithTwo', i, j, 5, 2))
    
    return answer

def isStraight(cards : List[str], level : int) -> Tuple[int, bool]:
    wild_card = getWildCard(level)
    card_nums = [0] * 16
    
    for card in cards:
        if card == wild_card:
            card_nums[-1] += 1
        rank = card[1]
        if rank.isnumeric():
            card_nums[int(rank) - 2] += 1
        else:
            card_nums[CardNumToNum[rank] - 2] += 1
    
    rank = -1
    
    for i in range(0, 10):
        start = 12 if i == 0 else i - 1
        if findStraightStartFrom(cards, start, 4, card_nums[-1]):
            rank = start + 1
            break
    
    if rank == -1:
        return (-1, False)
    return (rank, isStraightFlush(cards, level))

def isStraightFlush(cards : List[str], level : int):
    suit = None
    wild_card = getWildCard(level)
    for card in cards:
        if card != wild_card:
            if suit is None:
                suit = card[0]
            elif suit != card[0]:
                return False
    return True            

def findStraightStartFrom(cards : List[str], start : int, remain : int, wild_card_num : int) -> bool:
    if remain == 0:
        return cards[start] > 0 or wild_card_num > 0
    if cards[start] > 0:
        new_start = 0 if start == 12 else start + 1
        return findStraightStartFrom(cards, new_start, remain - 1, wild_card_num)
    elif wild_card_num > 0:
        new_start = 0 if start == 12 else start + 1
        return findStraightStartFrom(cards, new_start, remain - 1, wild_card_num - 1)
    else:
        return False

def findAllBomb(card_dict : Dict[str, int]) -> List[CardCombNoSuit]:
    answer = list()
    
    for size in [4, 5]:
        for i in range(1, 14):
            if card_dict[i] >= size:
                answer.append(CardCombNoSuit('Bomb', i, 0, size, 0))
            elif card_dict[i] == size - 1 and card_dict[16] >= 1:
                answer.append(CardCombNoSuit('Bomb', i, 0, size, 1))
            elif card_dict[i] == size - 2 and card_dict[16] == 2:
                answer.append(CardCombNoSuit('Bomb', i, 0, size, 2))
    
    if card_dict[14] == 2 and card_dict[15] == 2:
        answer.append(CardCombNoSuit('Bomb', 16, 0, 4, 0))
    
    return answer