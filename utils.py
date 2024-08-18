from typing import List, Dict, Final, Optional, Tuple, Union, Iterable, Callable
from time import time
from itertools import combinations, product
from numpy import random, ndarray as nplist

from cardcomb import CardComb, CombBase, isLargerThanRank

import copy

# random.seed(123123)

SUITS: Final[List[str]] = ['S', 'H', 'C', 'D']

POWERS: Final[List[str]] = ['2', '3', '4', '5', '6',
                            '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'B', 'R']

ScoreWeights : Final[List[Tuple[int]]] = [
    (1, 1), (2, 2), (3, 3), (5, 3), (6, 6), (6, 6), (5, 5), (100, 20), (460, 20), (560, 20), (1e7, 0)
]

TypeIndex : Final[Dict[str, int]] = {'Single' : 0, 'Pair' : 1, 'Trip' : 2, 'ThreeWithTwo' : 3, 
                                     'TwoTrips' : 4, 'ThreePairs' : 5, 'Straight' : 6, 'StraightFlush' : 7,
                                     'Bomb' : 8}

TypeNums : Final[List[Union[Tuple[str, int], Tuple[str, int, int]]]] = [('PASS', 1), 
                                                                        ('Single', 15), ('Pair', 15), ('Trip', 13),('ThreeWithTwo', 13),\
                                                                        ('TwoTrips', 13), ('ThreePairs', 12), ('Straight', 10),\
                                                                        ('Bomb', 4, 13), ('Bomb', 5, 13), ('StraightFlush', 10), ('Bomb', 6, 13),\
                                                                        ('Bomb', 7, 13), ('Bomb', 8, 13), ('Bomb', 9, 13), ('Bomb', 10, 13), ('Bomb', 11, 1)]

AllTypes : Final[int] = 223

CardNumToNum: Final[Dict[str, int]] = {
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, 'B': 15, 'R': 16
}

NumToCardNum: Final[Dict[int, str]] = {
    0: 'PASS', 1: 'A', 2: '2', 3: '3', 4 : '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: 'B', 16: 'R'
}

SuitToNum: Final[Dict[str, int]] = {
    'S': 0, 'H': 1, 'C': 2, 'D': 3
}

NumToSuit: Final[Dict[int, str]] = {
    0: 'S', 1: 'H', 2: 'C', 3: 'D'
}

def inRange(val : int, lower : int, upper : int) -> bool:
    '''
    Return whether val is in [lower, upper).
    '''
    return val >= lower and val < upper

def stageNum(num_of_card : int) -> int:
    if num_of_card >= 20:
        return 0
    elif num_of_card >= 12:
        return 1
    elif num_of_card >= 7:
        return 2
    else:
        return 9 - num_of_card

def getWildCard(level: int) -> str:
    if level < 1:
        level = 1
    elif level > 13:
        level = 13
    wild_card: str = 'H'
    if level < 9:
        wild_card += str(level + 1)
    else:
        wild_card += NumToCardNum[level + 1]
    return wild_card

def isLegalCard(card: str) -> bool:
    if len(card) != 2:
        return False
    if card in ['SB', 'HR']:
        return True
    suit: str = card[0]
    if suit not in SuitToNum.keys():
        return False
    else:
        number: str = card[1]
        if number.isnumeric():
            num = int(number)  # 2 - 9
            if num < 2 or num > 9:
                return False
        elif number not in CardNumToNum.keys():
            return False
        return True

def addCardToDict(cards: Union[str, List[str]], card_dict: Dict[str, int]) -> bool:
    if isinstance(cards, str):
        if isLegalCard(cards):
            card_dict[cards] += 1
            if card_dict[cards] <= 2:
                card_dict['total'] += 1
                return True
            else:
                card_dict[cards] = 2
                return False
        else:
            return False
    elif isinstance(cards, Iterable):
        all_fine = True
        for card in cards:
            if isinstance(card, str) and isLegalCard(card):
                card_dict[card] += 1
                if card_dict[card] > 2:
                    all_fine = False
                    card_dict[card] = 2
                else:
                    card_dict['total'] += 1
            else:
                all_fine = False
        return all_fine

def cardToNum(card: str) -> int:
    if isLegalCard(card):
        if card in ['SB', 'HR']:
            return 52 if card == 'SB' else 53
        suit: str = card[0]
        number: str = card[1]
        if number.isnumeric():
            num: int = int(number) - 2
        else:
            num: int = CardNumToNum[number] - 2
        return SuitToNum[suit] * 13 + num
    else:
        raise ValueError("The input card is illegal!")

def numOfCard(card: str) -> int:
    if not isLegalCard(card):
        return -1
    if card[1].isnumeric():
        return int(card[1])
    return CardNumToNum[card[1]]

def rankToInt(rank : str) -> int:
    if rank == 'JOKER':
        return -1
    return POWERS.index(rank)

def getCardDict(empty: bool = True) -> Dict[str, int]:
    card_dict = dict({})
    suits = list(SuitToNum.keys())
    for suit in suits:
        for i in range(2, 15):
            if i >= 10:
                c = suit + NumToCardNum[i]
                card_dict[c] = 0 if empty else 2
            else:
                c = suit + str(i)
                card_dict[c] = 0 if empty else 2
    card_dict['SB'] = 0 if empty else 2
    card_dict['HR'] = 0 if empty else 2
    card_dict['total'] = 0 if empty else 108
    return card_dict

def cardsToDict(cards: List[str], value_to_remove: Optional[int] = None) -> Dict[str, int]:
    card_dict = getCardDict()
    for card in cards:
        if isLegalCard(card):
            card_dict[card] += 1
            card_dict['total'] += 1
    if isinstance(value_to_remove, int):
        keys = list(card_dict.keys())
        for key in keys:
            if card_dict[key] == value_to_remove:
                _ = card_dict.pop(key)
    return card_dict

def cardDictToList(card_dict: Dict[str, int]) -> List[str]:
    cards = list()

    for num in POWERS:
        if num == 'B' and card_dict['SB'] > 0:
            cards.extend(['SB'] * card_dict['SB'])
        elif num == 'R' and card_dict['HR'] > 0:
            cards.extend(['HR'] * card_dict['HR'])
        elif num != 'B' and num != 'R':
            for suit in SUITS:
                card = suit + num
                card_num = card_dict[card]
                if card_num > 0:
                    cards.extend([card] * card_num)

    return cards

def cardDictToModelList(card_dict : Dict[str, int], level : int) -> List[int]:
    answer = [0] * 70
    
    for i in range(15):
        power = POWERS[i]
        if not power in ['B', 'R']:
            total = 0
            for suit in SUITS:
                card = f"{suit}{power}"
                answer[cardToNum(card)] = card_dict[card]
                total += card_dict[card]
            answer[54 + i] = total
        elif power == 'B':
            answer[52] = card_dict['SB']
        else:
            answer[53] = card_dict['HR']
    
    wild_card = getWildCard(level)
    
    answer[67] = card_dict[wild_card]
    answer[68] = card_dict['total']
    answer[69] = stageNum(card_dict['total'])
    
    return answer

def isLegalCardDict(card_dict: Dict[str, int]) -> bool:
    keys = list(card_dict.keys())
    try:
        assert keys.index('total') >= 0
    except AssertionError as e:
        print(e)
        return False
    total = 0
    for key in keys:
        total += card_dict[key] if key != 'total' else 0
    if total == card_dict['total']:
        return True
    else:
        print("The number of cards does not match the counter value!!!")
        return False

def randomCardDict(num: int) -> Dict[str, int]:
    if num > 108:
        num = 108
    elif num < 1:
        num = 1
    adding = True if num <= 54 else False
    card_dict = getCardDict(adding)
    while True:
        if card_dict['total'] == num:
            break
        card_index = random.randint(54)
        if card_index == 53:
            if adding and card_dict['HR'] < 2:
                card_dict['HR'] += 1
                card_dict['total'] += 1
            elif not adding and card_dict['HR'] > 0:
                card_dict['HR'] -= 1
                card_dict['total'] -= 1
        elif card_index == 52:
            if adding and card_dict['SB'] < 2:
                card_dict['SB'] += 1
                card_dict['total'] += 1
            elif not adding and card_dict['SB'] > 0:
                card_dict['SB'] -= 1
                card_dict['total'] -= 1
        else:
            suit = NumToSuit[int(card_index / 13)]
            card_num = NumToCardNum[card_index % 13 +
                                    1] if card_index % 13 == 0 or card_index % 13 >= 9 else str(card_index % 13 + 1)
            card = suit + card_num
            if adding and card_dict[card] < 2:
                card_dict[card] += 1
                card_dict['total'] += 1
            elif not adding and card_dict[card] > 0:
                card_dict[card] -= 1
                card_dict['total'] -= 1
    return card_dict

def generateNRandomCardLists(nums: Tuple) -> List[List[str]]:
    total_num = 0
    for num in nums:
        assert num <= 27 and num > 0
        total_num += num

    dummy_card_dict = randomCardDict(total_num)
    dummy_card_list = cardDictToList(dummy_card_dict)
    random.shuffle(dummy_card_list)

    lists = list()
    current_num = 0

    for num in nums:
        temp = dummy_card_list[current_num: current_num + num].copy()
        lists.append(temp)
        current_num += num

    return lists

def extractCardWithCardNum(card_dict: Dict[str, int], card_num: str) -> List[str]:
    cards = list()
    for suit in SUITS:
        card = suit + card_num
        if card_dict[card] == 2:
            cards.append(card)
            cards.append(card)
        elif card_dict[card] == 1:
            cards.append(card)
    return cards

def extractCardCombWith_N_WildCard(card_combs: List[List[str]], wild_card: str, num_wild_card: int) -> List[List[str]]:
    new_card_combs = list()
    for card_comb in card_combs:
        if card_comb.count(wild_card) == num_wild_card:
            new_card_combs.append(card_comb.copy())
    return new_card_combs

def separateCardCombByCardNum(card_combs: List[List[str]], power_A_first: bool = False) -> List[List[List]]:
    separated_card_combs = list()
    index = 0
    list_for_A = None

    for p in POWERS:
        if index >= len(card_combs):
            if p == 'A':
                list_for_A = list()
            separated_card_combs.append([])
        else:
            temp = list()
            while card_combs[index][0][1] == p:
                temp.append(card_combs[index].copy())
                index += 1
                if index >= len(card_combs):
                    break
            if p == 'A':
                list_for_A = temp.copy()
            separated_card_combs.append(temp)
    if power_A_first:
        separated_card_combs.insert(0, list_for_A)
    return separated_card_combs

def findAllCombs(card_dict: Dict[str, int], level: int) -> List[CardComb]:
    '''
    @param card_dict:\n
        This is the dict object that stores all the card(s) of the player.\n
        Please ensure the card dict object is legal before passing it into the function.\n
    @param level:\n
        If current level card is 2, then @param level is 1;\n
        If current level card is 7, then @param level is 6;\n
        If current level card is J, then @param level is 10;\n
        If current level card is A, then @param level is 13.\n
    '''
    if card_dict['total'] == 0:
        return list()

    wild_card = getWildCard(level)
    all_combs = list()
    num_card = card_dict['total']

    all_singles = findAllSingles(card_dict)
    for single in all_singles:
        new_single = CardComb('Single', numOfCard(single[0]) - 1, single)
        all_combs.append(new_single)

    if num_card >= 2:
        all_pairs = findAllPairs(card_dict, wild_card)
        for pair in all_pairs:
            new_pair = CardComb('Pair', numOfCard(pair[0]) - 1, pair)
            all_combs.append(new_pair)

    if num_card >= 3:
        all_trips = findAllTrips(card_dict, wild_card, all_singles, all_pairs)
        for trip in all_trips:
            new_trip = CardComb('Trip', numOfCard(trip[0]) - 1, trip)
            all_combs.append(new_trip)
    
    if num_card >= 5:
        all_three_with_twos = findAllThreeWithTwos(card_dict, wild_card, all_pairs, all_trips)
        for three_with_two in all_three_with_twos:
            new_three_with_two = CardComb('ThreeWithTwo', numOfCard(three_with_two[0]) - 1, three_with_two)
            all_combs.append(new_three_with_two)

    if num_card >= 6:
        all_three_pairs = findAllThreePairs(card_dict, wild_card, all_pairs)
        for three_pair in all_three_pairs:
            power = powerOfThreePairs(three_pair, wild_card)
            rank = 1 if power == 'A' else numOfCard(f"S{power}")
            new_three_pair = CardComb('ThreePairs', rank, three_pair)
            all_combs.append(new_three_pair)

        all_two_trips = findAllTwoTrips(card_dict, wild_card, all_trips)
        for two_trip in all_two_trips:
            rank = 1 if two_trip[0][1] == 'A' else numOfCard(two_trip[0])
            new_two_trip = CardComb('TwoTrips', rank, two_trip)
            all_combs.append(new_two_trip)

    if num_card >= 5:
        straights_and_straight_flushes = findAllStraightAndStraightFlush(card_dict, wild_card)
        all_straights = list()
        all_straightFlushes = list()

        for straight in straights_and_straight_flushes[0]:
            power = powerOfStraight(straight, wild_card)
            rank = 1 if power == 'A' else numOfCard(f"S{power}")
            all_straights.append(CardComb('Straight', rank, straight))
        all_combs.extend(all_straights)
        
        for straight in straights_and_straight_flushes[1]:
            power = powerOfStraight(straight, wild_card)
            rank = 1 if power == 'A' else numOfCard(f"S{power}")
            all_straightFlushes.append(CardComb('StraightFlush', rank, straight))
        all_combs.extend(all_straightFlushes)

    if num_card >= 4:
        all_bombs = findAllBombs(card_dict, wild_card)
        for bomb in all_bombs:
            new_bomb = CardComb('Bomb', numOfCard(bomb[0]) - 1, bomb)
            all_combs.append(new_bomb)

        joker_bomb = findJokerBomb(card_dict)
        if len(joker_bomb) == 1:
            new_joker_bomb = CardComb('Bomb', 16, bomb)
            all_combs.append(new_joker_bomb)

    return all_combs

def changeToCardComb(action : List[str], ctype : str) -> CardComb:
    if ctype == 'Single':
        return CardComb('Single', numOfCard(action[0]) - 1, action)
    if ctype == 'Pair':
        return CardComb('Pair', numOfCard(action[0]) - 1, action)

def changeToCardCombList(actions : List[List]) -> List[CardComb]:
    pass

def findAllSingles(card_dict: Dict[str, int]) -> List[List[str]]:
    singles = list()

    for num in POWERS:
        if num == 'B' and card_dict['SB'] > 0:
            singles.append(['SB'])
        elif num == 'R' and card_dict['HR'] > 0:
            singles.append(['HR'])
        elif num != 'B' and num != 'R':
            for suit in SUITS:
                card = suit + num
                if card_dict[card] > 0:
                    singles.append([card])

    return singles

def findAllPairs(card_dict: Dict[str, int], wild_card: str) -> List[List[str]]:
    pairs = list()
    num_wild_card = card_dict[wild_card]

    for i in range(2, 15):
        card_num = str(i) if i < 10 else NumToCardNum[i]
        cards = extractCardWithCardNum(card_dict, card_num)
        pair = list(set(combinations(cards, 2)))
        for p in pair:
            pairs.append(list(p))
        if wild_card[1] != card_num and num_wild_card > 0:
            for suit in SUITS:
                card = suit + card_num
                if card_dict[card] > 0:
                    pairs.append([card, wild_card])
    if card_dict['SB'] == 2:
        pairs.append(['SB', 'SB'])
    if card_dict['HR'] == 2:
        pairs.append(['HR', 'HR'])

    return pairs

def findAllTrips(card_dict: Dict[str, int], wild_card: str, singles: List[List[str]], pairs: List[List[str]]) -> List[List[str]]:
    trips = list()
    num_wild_card = card_dict[wild_card]

    for i in range(2, 15):
        cards = extractCardWithCardNum(
            card_dict, str(i) if i < 10 else NumToCardNum[i])
        trip = list(set(combinations(cards, 3)))
        for t in trip:
            trips.append(list(t))

    if num_wild_card >= 1:
        for pair in pairs:
            if pair.count(wild_card) == 0 and pair[0][1] != wild_card[1]:
                trips.append(pair + [wild_card])

    if num_wild_card == 2:
        for single in singles:
            if single[0][1] != wild_card[1]:
                trips.append(single + [wild_card, wild_card])

    return trips

def findAllThreePairs(card_dict: Dict[str, int], wild_card: str, pairs: List[List[str]]) -> List[List[str]]:
    '''
    0 wild cards:
    - AA2233
    1 wild card:
    - A*2233
    - AA2*33
    - AA223*
    2 wild cards:
    - A*2*33
    - A*223*
    - AA2*3*
    - **2233
    - AA**33
    - AA22**
    '''
    all_three_pairs = list()

    pairs_with_0_wild_card = extractCardCombWith_N_WildCard(
        pairs, wild_card, 0)
    separate_pairs_0_hlc = separateCardCombByCardNum(
        pairs_with_0_wild_card, True)

    pairs_with_1_wild_card = None
    separate_pairs_1_hlc = None

    for i in range(12):
        t1 = separate_pairs_0_hlc[i].copy()
        t2 = separate_pairs_0_hlc[i + 1].copy()
        t3 = separate_pairs_0_hlc[i + 2].copy()
        if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
            temp = list(product(t1, t2, t3))
            for prod in temp:
                threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                all_three_pairs.append(threepair)

    if card_dict[wild_card] >= 1:
        pairs_with_1_wild_card = extractCardCombWith_N_WildCard(
            pairs, wild_card, 1)
        separate_pairs_1_hlc = separateCardCombByCardNum(
            pairs_with_1_wild_card, True)
        # 1 wild card
        for i in range(14):
            if i > 1:
                t1 = separate_pairs_0_hlc[i - 2].copy()
                t2 = separate_pairs_0_hlc[i - 1].copy()
                t3 = separate_pairs_1_hlc[i].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i > 0 and i < 13:
                t1 = separate_pairs_0_hlc[i - 1].copy()
                t2 = separate_pairs_1_hlc[i].copy()
                t3 = separate_pairs_0_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i < 12:
                t1 = separate_pairs_1_hlc[i].copy()
                t2 = separate_pairs_0_hlc[i + 1].copy()
                t3 = separate_pairs_0_hlc[i + 2].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)

    if card_dict[wild_card] == 2:
        # 2 wild cards (in different pairs)
        for i in range(14):
            if i > 1:
                t1 = separate_pairs_1_hlc[i - 2].copy()
                t2 = separate_pairs_1_hlc[i - 1].copy()
                t3 = separate_pairs_0_hlc[i].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i > 0 and i < 13:
                t1 = separate_pairs_1_hlc[i - 1].copy()
                t2 = separate_pairs_0_hlc[i].copy()
                t3 = separate_pairs_1_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i < 12:
                t1 = separate_pairs_0_hlc[i].copy()
                t2 = separate_pairs_1_hlc[i + 1].copy()
                t3 = separate_pairs_1_hlc[i + 2].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)

        # 2 wild cards (in same pair)
        for i in range(14):
            if i > 1:
                t1 = separate_pairs_0_hlc[i - 2].copy()
                t2 = separate_pairs_0_hlc[i - 1].copy()
                if not (len(t1) == 0 or len(t2) == 0):
                    t3 = [[wild_card, wild_card]]
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i > 0 and i < 13:
                t1 = separate_pairs_0_hlc[i - 1].copy()
                t3 = separate_pairs_0_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t3) == 0):
                    t2 = [[wild_card, wild_card]]
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i < 12:
                t2 = separate_pairs_0_hlc[i + 1].copy()
                t3 = separate_pairs_0_hlc[i + 2].copy()
                if not (len(t2) == 0 or len(t3) == 0):
                    t1 = [[wild_card, wild_card]]
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() +
                                     prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)

    return all_three_pairs

def powerOfThreePairs(three_pair: List[str], wild_card: str) -> str:
    index = -1
    for index in range(6):
        if three_pair[index] != wild_card:
            break
    if index < 2:
        return three_pair[index][1]
    else:
        current_num = numOfCard(three_pair[index])
        first_num = current_num - 1
        if first_num == 1 or first_num >= 10:
            return NumToCardNum[first_num]
        else:
            return str(first_num)

def findAllTwoTrips(card_dict: Dict[str, int], wild_card: str, trips: List[List[str]]) -> List[List[str]]:
    '''
    0 wild cards:
    - AAA222
    1 wild card:
    - AA*222
    - AAA22*
    2 wild cards:
    - AA*22*
    - A**222
    - AAA2**
    '''
    all_two_trips = list()

    trips_with_0_wild_card = extractCardCombWith_N_WildCard(
        trips, wild_card, 0)
    separate_trips_0_hlc = separateCardCombByCardNum(
        trips_with_0_wild_card, True)

    # 0 wild cards
    for i in range(13):
        t1 = separate_trips_0_hlc[i].copy()
        t2 = separate_trips_0_hlc[i + 1].copy()
        if not (len(t1) == 0 or len(t2) == 0):
            temp = list(product(t1, t2))
            for prod in temp:
                two_trip = prod[0].copy() + prod[1].copy()
                all_two_trips.append(two_trip)

    if card_dict[wild_card] >= 1:
        trips_with_1_wild_card = extractCardCombWith_N_WildCard(
            trips, wild_card, 1)
        separate_trips_1_hlc = separateCardCombByCardNum(
            trips_with_1_wild_card, True)
        # 1 wild card
        for i in range(14):
            if i < 13:
                t1 = separate_trips_1_hlc[i].copy()
                t2 = separate_trips_0_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t2) == 0):
                    temp = list(product(t1, t2))
                    for prod in temp:
                        two_trip = prod[0].copy() + prod[1].copy()
                        all_two_trips.append(two_trip)
            if i > 0:
                t1 = separate_trips_0_hlc[i - 1].copy()
                t2 = separate_trips_1_hlc[i].copy()
                if not (len(t1) == 0 or len(t2) == 0):
                    temp = list(product(t1, t2))
                    for prod in temp:
                        two_trip = prod[0].copy() + prod[1].copy()
                        all_two_trips.append(two_trip)

    if card_dict[wild_card] == 2:
        trips_with_2_wild_card = extractCardCombWith_N_WildCard(
            trips, wild_card, 2)
        separate_trips_2_hlc = separateCardCombByCardNum(
            trips_with_2_wild_card, True)
        # 2 wild cards (in different trips)
        for i in range(13):
            t1 = separate_trips_1_hlc[i].copy()
            t2 = separate_trips_1_hlc[i + 1].copy()
            if not (len(t1) == 0 or len(t2) == 0):
                temp = list(product(t1, t2))
                for prod in temp:
                    two_trip = prod[0].copy() + prod[1].copy()
                    all_two_trips.append(two_trip)

        # 2 wild cards (in same trip)
        for i in range(14):
            if i < 13:
                t1 = separate_trips_2_hlc[i].copy()
                t2 = separate_trips_0_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t2) == 0):
                    temp = list(product(t1, t2))
                    for prod in temp:
                        two_trip = prod[0].copy() + prod[1].copy()
                        all_two_trips.append(two_trip)
            if i > 0:
                t1 = separate_trips_0_hlc[i - 1].copy()
                t2 = separate_trips_2_hlc[i].copy()
                if not (len(t1) == 0 or len(t2) == 0):
                    temp = list(product(t1, t2))
                    for prod in temp:
                        two_trip = prod[0].copy() + prod[1].copy()
                        all_two_trips.append(two_trip)

    return all_two_trips

def findAllThreeWithTwos(card_dict: Dict[str, int], wild_card: str, pairs: List[List[str]], trips: List[List[str]]) -> List[List[str]]:
    '''
    0 wild cards:
    - AAA22
    1 wild card:
    - AA*22
    - AAA2* (No SB and HR)
    2 wild cards:
    - A**22
    - AA*2* (No SB and HR)
    '''
    all_three_with_twos = list()

    pairs_with_0_wild_card = extractCardCombWith_N_WildCard(
        pairs, wild_card, 0)
    separate_pairs_0_hlc = separateCardCombByCardNum(
        pairs_with_0_wild_card)
    pairs_with_1_wild_card = None
    separate_pairs_1_hlc = None

    trips_with_0_wild_card = extractCardCombWith_N_WildCard(
        trips, wild_card, 0)
    separate_trips_0_hlc = separateCardCombByCardNum(
        trips_with_0_wild_card)
    trips_with_1_wild_card = None
    separate_trips_1_hlc = None
    trips_with_2_wild_card = None
    separate_trips_2_hlc = None

    # 0 wild cards:
    for i in range(0, 13):
        for j in range(0, 13):
            if i != j:
                trip = separate_trips_0_hlc[i]
                pair = separate_pairs_0_hlc[j]
                temp = list(product(trip, pair))
                for prod in temp:
                    three_with_two = prod[0].copy() + prod[1].copy()
                    all_three_with_twos.append(three_with_two)

    if card_dict[wild_card] >= 1:
        pairs_with_1_wild_card = extractCardCombWith_N_WildCard(
            pairs, wild_card, 1)
        separate_pairs_1_hlc = separateCardCombByCardNum(
            pairs_with_1_wild_card)
        trips_with_1_wild_card = extractCardCombWith_N_WildCard(
            trips, wild_card, 1)
        separate_trips_1_hlc = separateCardCombByCardNum(
            trips_with_1_wild_card)
        # 1 wild cards:
        for i in range(0, 13):
            for j in range(0, 13):
                if i != j:
                    trip1 = separate_trips_1_hlc[i]
                    pair1 = separate_pairs_0_hlc[j]
                    temp1 = list(product(trip1, pair1))
                    for prod in temp1:
                        three_with_two = prod[0].copy() + prod[1].copy()
                        all_three_with_twos.append(three_with_two)
                    trip2 = separate_trips_0_hlc[i]
                    pair2 = separate_pairs_1_hlc[j]
                    temp2 = list(product(trip2, pair2))
                    for prod in temp2:
                        three_with_two = prod[0].copy() + prod[1].copy()
                        all_three_with_twos.append(three_with_two)

    if card_dict[wild_card] == 2:
        trips_with_2_wild_card = extractCardCombWith_N_WildCard(
            trips, wild_card, 2)
        separate_trips_2_hlc = separateCardCombByCardNum(
            trips_with_2_wild_card)
        # 2 wild cards:
        for i in range(0, 13):
            for j in range(0, 13):
                if i != j:
                    trip1 = separate_trips_2_hlc[i]
                    pair1 = separate_pairs_0_hlc[j]
                    temp1 = list(product(trip1, pair1))
                    for prod in temp1:
                        three_with_two = prod[0].copy() + prod[1].copy()
                        all_three_with_twos.append(three_with_two)
                    trip2 = separate_trips_1_hlc[i]
                    pair2 = separate_pairs_1_hlc[j]
                    temp2 = list(product(trip2, pair2))
                    for prod in temp2:
                        three_with_two = prod[0].copy() + prod[1].copy()
                        all_three_with_twos.append(three_with_two)

    return all_three_with_twos

def findAllBombs(card_dict: Dict[str, int], wild_card: str) -> List[List[str]]:
    allBombs = list()
    num_wild_card = card_dict[wild_card]
    bomb_size_upper = 8 + num_wild_card
    for s in range(4, bomb_size_upper + 1):
        bombs = findAllBombsWithSize(
            card_dict, s, wild_card, num_wild_card)
        allBombs.extend(bombs)
    return allBombs

def findAllBombsWithSize(card_dict: Dict[str, int], size: int, wild_card: str, num_wild_card: int) -> List[List[str]]:
    '''
    This function does not find JOKER Bomb.
    '''
    bombs = list()

    for i in range(2, 15):
        raw_cards = extractCardWithCardNum(
            card_dict, str(i) if i < 10 else NumToCardNum[i])
        cards = []
        for card in raw_cards:
            if card != wild_card:
                cards.append(card)
        if len(cards) == 0 or len(cards) < size - num_wild_card:
            continue
        if len(cards) == size - num_wild_card and num_wild_card > 0:
            extra = [wild_card, wild_card]
            if num_wild_card == 1:
                _ = extra.pop()
                assert len(extra) == 1
            bombs.append((cards + extra).copy())
        else:
            temp = list(combinations(cards, size))
            for b in temp:
                bombs.append(list(b))
            if num_wild_card >= 1:
                temp2 = list(combinations(cards, size - 1))
                for b2 in temp2:
                    bombs.append(list(b2) + [wild_card])
            if num_wild_card >= 2:
                temp3 = list(combinations(cards, size - 2))
                for b3 in temp3:
                    bombs.append(
                        list(b3) + [wild_card, wild_card])

    return bombs

def findJokerBomb(card_dict: Dict[str, int]) -> List[List[str]]:
    answer = list()

    if card_dict['SB'] == 2 and card_dict['HR'] == 2:
        answer.append(['SB', 'SB', 'HR', 'HR'])

    return answer

def findAllStraightAndStraightFlush(card_dict: Dict[str, int], wild_card: str) -> List[List[List[str]]]:
    answer = list()
    temp = findAllStraights(card_dict, wild_card)
    temp2 = findAllStraightFlushes(card_dict, wild_card)
    answer.append(temp)
    answer.append(temp2)
    return answer

def findAllStraights(card_dict: Dict[str, int], wild_card: str) -> List[List[str]]:
    allStraight = list()
    num_heart_level = card_dict[wild_card]
    for i in range(1, 11):
        straights = findAllStraightFrom(
            card_dict, wild_card, i, 4, num_heart_level)
        if len(straights) > 0:
            for straight in straights:
                if not isStraightFlush(straight, wild_card)[0]:
                    allStraight.append(list(straight))
    return allStraight

def findAllStraightFlushes(card_dict: Dict[str, int], wild_card: str) -> List[List[str]]:
    allStraightFlush = list()
    num_heart_level = card_dict[wild_card]
    for i in range(1, 11):
        for suit in SUITS:
            straights = findAllStraightFlushFrom(
                card_dict, suit, wild_card, i, 4, num_heart_level)
            if len(straights) > 0:
                for straight in straights:
                    allStraightFlush.append(list(straight))
    return allStraightFlush

def findAllStraightFrom(card_dict: Dict[str, int], wild_card: str, current: int, remain: int, num_wild_card: int) -> List[List[str]]:
    card_num = NumToCardNum[current] if current == 1 or current >= 10 else str(
        current)
    answer = list()
    use_wild_card = False
    for suit in SUITS:
        card = suit + card_num
        if card_dict[card] > 0:
            if remain == 0:
                if card == wild_card and num_wild_card > 0:
                    use_wild_card = True
                    answer.append([card])
                elif card != wild_card:
                    answer.append([card])
            else:
                next_straight = list()
                if card == wild_card and num_wild_card > 0:
                    use_wild_card = True
                    next_straight = findAllStraightFrom(
                        card_dict, wild_card, current + 1, remain - 1, num_wild_card - 1)
                elif card != wild_card:
                    next_straight = findAllStraightFrom(
                        card_dict, wild_card, current + 1, remain - 1, num_wild_card)
                if len(next_straight) > 0:
                    for data in next_straight:
                        t = [card] + list(data)
                        answer.append(t)
    if not use_wild_card and num_wild_card > 0:
        if remain == 0:
            answer.append([wild_card])
        else:
            next_straight = findAllStraightFrom(
                card_dict, wild_card, current + 1, remain - 1, num_wild_card - 1)
            if len(next_straight) > 0:
                for data in next_straight:
                    t = [wild_card] + list(data)
                    answer.append(t)
    return answer

def findAllStraightFlushFrom(card_dict: Dict[str, int], suit : str, wild_card: str, current: int, remain: int, num_wild_card: int) -> List[List[str]]:
    card_num = NumToCardNum[current] if current == 1 or current >= 10 else str(current)
    answer = list()
    use_wild_card = False
    card = suit + card_num
    if card_dict[card] > 0:
        if remain == 0:
            if card == wild_card and num_wild_card > 0:
                use_wild_card = True
                answer.append([card])
            elif card != wild_card:
                answer.append([card])
        else:
            next_straight = list()
            if card == wild_card and num_wild_card > 0:
                use_wild_card = True
                next_straight = findAllStraightFlushFrom(
                    card_dict, suit, wild_card, current + 1, remain - 1, num_wild_card - 1)
            elif card != wild_card:
                next_straight = findAllStraightFlushFrom(
                    card_dict, suit, wild_card, current + 1, remain - 1, num_wild_card)
            if len(next_straight) > 0:
                for data in next_straight:
                    t = [card] + list(data)
                    answer.append(t)
    if not use_wild_card and num_wild_card > 0:
        if remain == 0:
            answer.append([wild_card])
        else:
            next_straight = findAllStraightFlushFrom(
                card_dict, suit, wild_card, current + 1, remain - 1, num_wild_card - 1)
            if len(next_straight) > 0:
                for data in next_straight:
                    t = [wild_card] + list(data)
                    answer.append(t)
    return answer

def isStraightFlush(cardComb: Iterable[str], wild_card: str) -> Tuple[bool, Optional[str]]:
    '''
    @param cardComb
    This is the Straight Combination that we are going to test.
    We assume it is a legal Straight Combination.
    '''
    if len(cardComb) != 5:
        return (False, None)
    suit = 'None'
    for card in cardComb:
        if card != wild_card:
            if suit not in SUITS:
                suit = card[0]
            elif suit in SUITS and card[0] != suit:
                return (False, None)
    return (True, suit)

def powerOfStraight(straight: Iterable[str], wild_card: str) -> str:
    p = 'None'
    num = -1
    for i in range(3):
        if straight[i] != wild_card:
            num = CardNumToNum[straight[i][1]] % 13 if not straight[i][1].isnumeric(
            ) else int(straight[i][1])
            num -= i
    p = NumToCardNum[num] if num == 1 or num == 10 else str(num)
    return p

def suitOfStraightFlush(straight_flush : Iterable[str], wild_card : str) -> str:
    suit = None
    for card in straight_flush:
        if card == wild_card:
            continue
        else:
            suit = card[0]
            break
    if suit == None:
        raise ValueError("The @param straight_flush is invalid!")
    return suit

def updateCardDictAfterAction(card_dict: Dict[str, int], action: Optional[CardComb]) -> Dict[str, int]:
    if action == None:
        return card_dict.copy()

    answer = card_dict.copy()
    for card in action.cards:
        answer[card] -= 1
        answer['total'] -= 1

    return answer

def updateCardCombsAfterAction(card_combs: List[CardComb], card_dict: Dict[str, int], action: Optional[CardComb]) -> Tuple[List[CardComb], List[CardComb]]:
    if action == None:
        return (list(), card_combs.copy())

    contain_action = any(action == c for c in card_combs)

    if not contain_action:
        return (list(), card_combs.copy())

    action_dict = cardsToDict(action[2], 0)
    _ = action_dict.pop('total')
    keys = list(action_dict.keys())

    for key in keys:
        card_dict[key] -= action_dict[key]
        card_dict['total'] -= action_dict[key]

    new_combs = list()
    fail_combs = list()

    for comb in card_combs:
        can_be_added = True
        for key in keys:
            if comb.cards.count(key) == 2:
                can_be_added = False
            elif comb.cards.count(key) == 1:
                if action_dict[key] == 2 or (action_dict[key] == 1 and card_dict[key] == 0):
                    can_be_added = False
            if not can_be_added:
                break
        if can_be_added:
            new_combs.append(comb)
        else:
            fail_combs.append(comb)

    return fail_combs, new_combs

def isSameAction(action1: Optional[CardComb], action2: Optional[CardComb]) -> bool:
    if action1 == None or action2 == None:
        return action1 == None and action2 == None
    return action1 == action2

def filterActions(actions: List[CardComb], base_action: Optional[CardComb], level: int) -> List[CardComb]:
    if base_action is None or base_action.is_pass():
        return actions
    elif base_action.t == 'Bomb' and base_action.rank == 16:
        return [CardComb.pass_cardcomb()]
    elif not base_action.t in ['Bomb', 'StraightFlush']:
        final_actions = list()
        final_actions.append(CardComb.pass_cardcomb())
        base_rank = base_action.rank
        for action in actions:
            temp = action.t
            if temp in ['Bomb', 'StraightFlush']:
                final_actions.append(action)
            elif temp == base_action.t:
                action_rank = action.rank
                if temp in ['ThreePairs', 'TwoTrips', 'Straight'] and isLargerThanRank(action_rank, base_rank, None):
                    final_actions.append(action)
                elif isLargerThanRank(action_rank, base_rank, level):
                    final_actions.append(action)
        return final_actions
    elif base_action.t == 'StraightFlush':
        final_actions = list()
        final_actions.append(CardComb.pass_cardcomb())
        base_rank = base_action.rank
        for action in actions:
            temp = action.t
            if temp == 'Bomb' and (action.rank == 16 or len(action.cards) >= 6):
                final_actions.append(action)
            elif temp == 'StraightFlush':
                action_rank = action.rank
                if isLargerThanRank(action_rank, base_rank, None):
                    final_actions.append(action)
        return final_actions
    else:
        final_actions = list()
        final_actions.append(CardComb.pass_cardcomb())
        base_size = len(base_action.cards)
        base_rank = base_action.rank
        for action in actions:
            temp = action.t
            if temp == 'StraightFlush':
                if base_size <= 5:
                    final_actions.append(action)
            elif temp == 'Bomb':
                if action.rank == 16 or len(action.cards) > base_size:
                    final_actions.append(action)
                elif len(action.cards) == base_size:
                    action_rank = action.rank
                    if isLargerThanRank(action_rank, base_rank, level):
                        final_actions.append(action)
        return final_actions

def passIsFine(oppo_actions: List[CardComb], oppo_card_num: int) -> bool:
    '''
    This function returns true if opponent cannot play all the card(s) in one time;\n
    Else, it returns False.
    '''
    for i in range(len(oppo_actions) - 1, -1, -1):
        if not oppo_actions[i].is_pass() and len(oppo_actions[i].cards) == oppo_card_num:
            return False
    return True

def canPassOnly(current_action: Optional[CardComb], oppo_actions: List[CardComb], level: int) -> bool:
    if current_action is None:
        return False
    filtered_actions = filterActions(oppo_actions, current_action, level)
    for action in filtered_actions:
        if not action.is_pass():
            return False
    return True

def canPlayAllInOnce(actions: List[CardComb], num_card: int) -> int:
    index = -1
    for i in range(len(actions) - 1, -1, -1):
        if not actions[i].is_pass() and len(actions[i].cards) == num_card:
            index = i
            break
    return index

def getFlagsForActions(all_combs : List[CardComb], base_action : CardComb, level : int, consider_suit_for_sf : bool = False) -> List[List[int]]:
    
    def init_flags(flags : List) -> None:
        size = 19 if consider_suit_for_sf else 16
        for _ in range(size):
            flags.append([0] * 15)
    
    flags = []
    init_flags(flags)
    size = 0
    value = 0
    base = 3 if consider_suit_for_sf else 0
    wild_card = getWildCard(level)
    
    for comb in all_combs:
        value = 2 if CombBase.actionComparision(comb, base_action, level) else 1
        type_index = TypeIndex[comb.t]
        if type_index == 8:
            if comb.rank == 16:
                for i in range(15):
                    flags[15 + base][i] = 2
            else:
                size = len(comb.cards)
                if size < 6:
                    flags[3 + size][comb.rank - 1] = value
                else:
                    flags[4 + size + base][comb.rank - 1] = value
        if type_index <= 7 and type_index >= 4:
            if type_index == 7:
                if consider_suit_for_sf:
                    suit = suitOfStraightFlush(comb.cards, wild_card)
                    flags[9 + SUITS.index(suit)][comb.rank - 1] = value
                else:
                    flags[9][comb.rank - 1] = value
            else:
                flags[type_index + base][comb.rank - 1] = value
        else:
            flags[type_index + base][comb.rank] = value
    return flags

def getFlagsForStraightFlush(card_dict : Dict[str, int], action : CardComb, wild_card : str) -> List[List[int]]:
    temp_dict = copy.copy(card_dict)
    
    for card in action.cards:
        temp_dict[card] -= 1
        temp_dict['total'] -= 1
    
    flags = []
    
    for _ in range(4):
        flags.append([0] * 10)
    
    if temp_dict['total'] < 5:
        return flags
    
    sfs = findAllStraightFlushes(temp_dict, wild_card)
    
    if len(sfs) > 0:
        for sf in sfs:
            suit = suitOfStraightFlush(sf, wild_card)
            power = powerOfStraight(sf, wild_card)
            rank = 1 if power == 'A' else numOfCard(f"S{power}")
            flags[SUITS.index(suit) * 10][rank - 1] = 1
    
    return flags

def getChoiceUnderThreeWithTwo(card_dict : Dict[str, int], actions : List[CardComb], wild_card : str) -> CardComb:
    '''
    This function returns the best choice given the action base is ThreeWithTwo.
    
    Procedure:
    Calculate the final score after choosing each action, and return the action with highest score.
    The weight Matrix is still in progress, but you can use @var ScoreWeights before the update.
    '''

def getChoiceUnderAction(card_dict : Dict[str, int], actions : List[CardComb], wild_card : str) -> CardComb:
    answer : Optional[CardComb] = None
    lowest_wild_card_num : int = 3
    highest_score : int = -1
    for action in actions:
        flags_sf = getFlagsForStraightFlush(card_dict, action, wild_card)
        score = scoreOfSraightFlushFlags(flags_sf)
        wcn = action.num_wild_card(wild_card)
        if wcn < lowest_wild_card_num:
            answer = action
            highest_score = score
        elif wcn == lowest_wild_card_num:
            if score > highest_score:
                answer = action
                highest_score = score
    return answer

def scoreOfSraightFlushFlags(flags : List[List[int]]) -> int:
    score = 0
    
    for flag in flags:
        for i in range(10):
            score += flag[i] * (i + 1)
    
    return score

def checkCardCombTypeRank(cardcomb : CardComb, base : Union[CardComb, CombBase]) -> bool:
    if isinstance(base, CardComb):
        return cardcomb.t == base.t and cardcomb.rank == base.rank and len(cardcomb.cards) == len(base.cards)
    else:
        return cardcomb.t == base.t and cardcomb.rank == base.rank and len(cardcomb.cards) == base.cards_num

def assignCombBaseToProbs(probs : List[float], indices : List[int], base_action : CombBase, level : int) -> List[Tuple[CombBase, float]]:
    answer = list()
    for i in range(194):
        if i == 0 and not base_action.is_pass():
            answer.append((CombBase.pass_comb(), probs[i]))
        elif indices[i] == 1:
            comb = indexToCombBase(i)
            if CombBase.actionComparision(comb, base_action, level):
                answer.append((comb, probs[i]))
    return answer

def indexOfCombBase(comb : CombBase) -> int:
    if comb.is_pass():
        return 0
    elif comb.is_joker_bomb():
        return 193
    index = TypeIndex[comb.t]
    base = 0
    if index < 7:
        base = sum(data[1] for data in TypeNums[0:index + 1])
    elif index == 7:
        base = 118
    else:
        size = comb.cards_num
        if size < 6:
            base = 92 + (size - 4) * 13
        else:
            base = 128 + (size - 6) * 13
    return base + comb.rank - 1

def indexToCombBase(index : int) -> CombBase:
    if index == 0:
        return CombBase.pass_comb()
    if index == 193:
        return CombBase.jokerbomb_comb()
    if index < 16:
        return CombBase('Single', index, 1)
    if index < 31:
        return CombBase('Pair', index - 15, 2)
    if index < 44:
        return CombBase('Trip', index - 30, 3)
    if index < 57:
        return CombBase('ThreeWithTwo', index - 43, 5)
    if index < 70:
        return CombBase('TwoTrips', index - 56, 6)
    if index < 82:
        return CombBase('ThreePairs', index - 69, 6)
    if index < 92:
        return CombBase('Straight', index - 81, 5)
    if index < 105:
        return CombBase('Bomb', index - 91, 4)
    if index < 118:
        return CombBase('Bomb', index - 104, 5)
    if index < 128:
        return CombBase('StraightFlush', index - 117, 5)
    if index < 141:
        return CombBase('Bomb', index - 127, 6)
    if index < 154:
        return CombBase('Bomb', index - 140, 7)
    if index < 167:
        return CombBase('Bomb', index - 153, 8)
    if index < 180:
        return CombBase('Bomb', index - 166, 9)
    if index < 193:
        return CombBase('Bomb', index - 179, 10)
    raise ValueError("@param index is invalid!")

if __name__ == "__main__":
    index = 188
    print(indexToCombBase(index))
