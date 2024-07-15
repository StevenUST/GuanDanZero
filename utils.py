from typing import List, Dict, Final, Optional, Tuple
from time import time
from itertools import combinations, product
from numpy import random

#random.seed(123123)

SUITS : Final[List[str]] = ['S', 'H', 'C', 'D']

POWERS : Final[List[str]] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'B', 'R']

CardNumToNum : Final[Dict[str, int]] = {
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

NumToCardNum : Final[Dict[int, str]] = {
    1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
}

SuitToNum : Final[Dict[str, int]] = {
    'S': 0, 'H': 1, 'C': 2, 'D': 3
}

NumToSuit : Final[Dict[int, str]] = {
    0: 'S', 1: 'H', 2: 'C', 3: 'D'
}

def inRange(val : int, bound : Tuple[int, int]) -> bool:
    return val <= bound[1] and val >= bound[0]

def getHeartLevelCard(level : int) -> str:
    if level < 1:
        level = 1
    elif level > 13:
        level = 13
    heart_level_card : str = 'H'
    if level < 9:
        heart_level_card += str(level + 1)
    else:
        heart_level_card += NumToCardNum[level + 1]
    return heart_level_card

def isLegalCard(card : str) -> bool:
    if len(card) != 2:
        return False
    if card in ['SB', 'HR']:
        return True
    suit : str = card[0]
    if suit not in SuitToNum.keys():
        return False
    else:
        number : str = card[1]
        if number.isnumeric():
            num = int(number) # 2 - 9
            if num < 2 or num > 9:
                return False
        elif number not in CardNumToNum.keys():
           return False
        return True

def cardToNum(card : str) -> int:
    if isLegalCard(card):
        if card in ['SB', 'HR']:
            return 52 if card == 'SB' else 53
        suit : str = card[0]
        number : str = card[1]
        if number.isnumeric():
            num : int = int(number) - 2
        else:
            num : int = CardNumToNum[number] - 2
        return SuitToNum[suit] * 13 + num
    else:
        raise ValueError("The input card is illegal!")

def numOfCard(card : str) -> int:
    if not isLegalCard(card):
        return -1
    if card[1].isnumeric():
        return int(card[1])
    return CardNumToNum[card[1]]

def getCardDict(empty : bool = True) -> Dict[str, int]:
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
    
def cardsToDict(cards : List[str], value_to_remove : Optional[int] = None) -> Dict[str, int]:
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

def cardDictToList(card_dict : Dict[str, int]) -> List[str]:
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

def isLegalCardDict(card_dict : Dict[str, int]) -> bool:
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

def numOfWildCard(card_dict : Dict[str, int], level : int) -> List:
    """
    return the number of wild card in the hand cards.
    """
    return card_dict[getHeartLevelCard(level)]

def randomCardDict(num : int) -> Dict[str, int]:
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
            card_num = NumToCardNum[card_index % 13 + 1] if card_index % 13 == 0 or card_index % 13 >= 9 else str(card_index % 13 + 1)
            card = suit + card_num
            if adding and card_dict[card] < 2:
                card_dict[card] += 1
                card_dict['total'] += 1
            elif not adding and card_dict[card] > 0:
                card_dict[card] -= 1
                card_dict['total'] -= 1
    return card_dict

def generateNRandomCardLists(nums : Tuple) -> List[List[str]]:
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
        temp = dummy_card_list[current_num : current_num + num].copy()
        lists.append(temp)
        current_num += num
    
    return lists

def extractCardWithCardNum(card_dict : Dict[str, int], card_num: str) -> List[str]:
    cards = list()
    for suit in SUITS:
        card = suit + card_num
        if card_dict[card] == 2:
            cards.append(card)
            cards.append(card)
        elif card_dict[card] == 1:
            cards.append(card)
    return cards

def extractCardCombWith_N_HeartLevelCard(card_combs : List[List[str]], heart_level_card : str, num_heart_level_card : int) -> List[List[str]]:
    new_card_combs = list()
    for card_comb in card_combs:
        if card_comb.count(heart_level_card) == num_heart_level_card:
            new_card_combs.append(card_comb.copy())
    return new_card_combs

def separateCardCombByCardNum(card_combs : List[List[str]], power_A_first : bool = False) -> List[List[List]]:    
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

def findAllCombs(card_dict : Dict[str, int], level : int) -> List[List]:
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
    heart_level_card = getHeartLevelCard(level)
    all_combs = list()
    
    all_singles = findAllSingles(card_dict)
    for single in all_singles:
        new_single = ['Single', single[0][1], single]
        all_combs.append(new_single)
    
    all_pairs = findAllPairs(card_dict, heart_level_card)
    for pair in all_pairs:
        new_pair = ['Pair', pair[0][1], pair]
        all_combs.append(new_pair)
    
    all_trips = findAllTrips(card_dict, heart_level_card, all_singles, all_pairs)
    for trip in all_trips:
        new_trip = ['Trip', trip[0][1], trip]
        all_combs.append(new_trip)
    
    all_three_pairs = findAllThreePairs(card_dict, heart_level_card, all_pairs)
    for three_pair in all_three_pairs:
        new_three_pair = ['ThreePairs', powerOfThreePairs(three_pair, heart_level_card), three_pair]
        all_combs.append(new_three_pair)
    
    all_two_trips = findAllTwoTrips(card_dict, heart_level_card, all_trips)
    for two_trip in all_two_trips:
        new_two_trip = ['TwoTrips', two_trip[0][1], two_trip]
        all_combs.append(new_two_trip)
    
    all_three_with_twos = findAllThreeWithTwos(card_dict, heart_level_card, all_pairs, all_trips)
    for three_with_two in all_three_with_twos:
        new_three_with_two = ['ThreeWithTwo', three_with_two[0][1], three_with_two]
        all_combs.append(new_three_with_two)
    
    straights_and_straight_flushes = findAllStraightAndStraightFlush(card_dict, heart_level_card)
    all_straights = list()
    all_straightFlushes = list()
    for straight in straights_and_straight_flushes:
        if isStraightFlush(straight, heart_level_card):
            all_straightFlushes.append(['StraightFlush', powerOfStraight(straight, heart_level_card), straight])
        else:
            all_straights.append(['Straight', powerOfStraight(straight, heart_level_card), straight])

    all_combs.extend(all_straights)
    
    all_bombs = findAllBombs(card_dict, heart_level_card)
    for bomb in all_bombs:
        new_bomb = ['Bomb', bomb[0][1], bomb]
        all_combs.append(new_bomb)
    
    all_combs.extend(all_straightFlushes)
    
    joker_bomb = findJokerBomb(card_dict)
    if len(joker_bomb) == 1:
        new_joker_bomb = ['Bomb', 'JOKER', joker_bomb[0]]
        all_combs.append(new_joker_bomb)

    return all_combs

def findAllSingles(card_dict : Dict[str, int]) -> List[List[str]]:
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

def findAllPairs(card_dict : Dict[str, int], heart_level_card : str) -> List[List[str]]:
    pairs = list()
    num_heart_level_card = card_dict[heart_level_card]
    
    for i in range(2, 15):
        card_num = str(i) if i < 10 else NumToCardNum[i]
        cards = extractCardWithCardNum(card_dict, card_num)
        pair = list(set(combinations(cards, 2)))
        for p in pair:
            pairs.append(list(p))
        if heart_level_card[1] != card_num and num_heart_level_card > 0:
            for suit in SUITS:
                card = suit + card_num
                if card_dict[card] > 0:
                    pairs.append([card, heart_level_card])
    if card_dict['SB'] == 2:
        pairs.append(['SB', 'SB'])
    if card_dict['HR'] == 2:
        pairs.append(['HR', 'HR'])

    return pairs

def findAllTrips(card_dict : Dict[str, int], heart_level_card : str, singles : List[List[str]], pairs: List[List[str]]) -> List[List[str]]:
    trips = list()
    num_heart_level_card = card_dict[heart_level_card]
    
    for i in range(2, 15):
        cards = extractCardWithCardNum(card_dict, str(i) if i < 10 else NumToCardNum[i])
        trip = list(set(combinations(cards, 3)))
        for t in trip:
            trips.append(list(t))
    
    if num_heart_level_card >= 1:
        for pair in pairs:
            if pair.count(heart_level_card) == 0 and pair[0][1] != heart_level_card[1]:
                trips.append(pair + [heart_level_card])
    
    if num_heart_level_card == 2:
        for single in singles:
            if single[0][1] != heart_level_card[1]:
                trips.append(single + [heart_level_card, heart_level_card])
    
    return trips

def findAllThreePairs(card_dict : Dict[str, int], heart_level_card : str, pairs : List[List[str]]) -> List[List[str]]:
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
    
    pairs_with_0_heart_level_card = extractCardCombWith_N_HeartLevelCard(pairs, heart_level_card, 0)
    separate_pairs_0_hlc = separateCardCombByCardNum(pairs_with_0_heart_level_card, True)
    
    pairs_with_1_heart_level_card = None
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

    if card_dict[heart_level_card] >= 1:
        pairs_with_1_heart_level_card = extractCardCombWith_N_HeartLevelCard(pairs, heart_level_card, 1)
        separate_pairs_1_hlc = separateCardCombByCardNum(pairs_with_1_heart_level_card, True)
        # 1 wild card
        for i in range(14):
            if i > 1:
                t1 = separate_pairs_0_hlc[i - 2].copy()
                t2 = separate_pairs_0_hlc[i - 1].copy()
                t3 = separate_pairs_1_hlc[i].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i > 0 and i < 13:
                t1 = separate_pairs_0_hlc[i - 1].copy()
                t2 = separate_pairs_1_hlc[i].copy()
                t3 = separate_pairs_0_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i < 12:
                t1 = separate_pairs_1_hlc[i].copy()
                t2 = separate_pairs_0_hlc[i + 1].copy()
                t3 = separate_pairs_0_hlc[i + 2].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)

    if card_dict[heart_level_card] == 2:
        # 2 wild cards (in different pairs)
        for i in range(14):
            if i > 1:
                t1 = separate_pairs_1_hlc[i - 2].copy()
                t2 = separate_pairs_1_hlc[i - 1].copy()
                t3 = separate_pairs_0_hlc[i].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i > 0 and i < 13:
                t1 = separate_pairs_1_hlc[i - 1].copy()
                t2 = separate_pairs_0_hlc[i].copy()
                t3 = separate_pairs_1_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i < 12:
                t1 = separate_pairs_0_hlc[i].copy()
                t2 = separate_pairs_1_hlc[i + 1].copy()
                t3 = separate_pairs_1_hlc[i + 2].copy()
                if not (len(t1) == 0 or len(t2) == 0 or len(t3) == 0):
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
    
        # 2 wild cards (in same pair)
        for i in range(14):
            if i > 1:
                t1 = separate_pairs_0_hlc[i - 2].copy()
                t2 = separate_pairs_0_hlc[i - 1].copy()
                if not (len(t1) == 0 or len(t2) == 0):
                    t3 = [[heart_level_card, heart_level_card]]
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i > 0 and i < 13:
                t1 = separate_pairs_0_hlc[i - 1].copy()
                t3 = separate_pairs_0_hlc[i + 1].copy()
                if not (len(t1) == 0 or len(t3) == 0):
                    t2 = [[heart_level_card, heart_level_card]]
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
            if i < 12:
                t2 = separate_pairs_0_hlc[i + 1].copy()
                t3 = separate_pairs_0_hlc[i + 2].copy()
                if not (len(t2) == 0 or len(t3) == 0):
                    t1 = [[heart_level_card, heart_level_card]]
                    temp = list(product(t1, t2, t3))
                    for prod in temp:
                        threepair = (prod[0].copy() + prod[1].copy()) + prod[2].copy()
                        all_three_pairs.append(threepair)
    
    return all_three_pairs

def powerOfThreePairs(three_pair : List[str], heart_level_card : str) -> str:
    index = -1
    for index in range(6):
        if three_pair[index] != heart_level_card:
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

def findAllTwoTrips(card_dict : Dict[str, int], heart_level_card : str, trips : List[List[str]]) -> List[List[str]]:
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
    
    trips_with_0_heart_level_card = extractCardCombWith_N_HeartLevelCard(trips, heart_level_card, 0)
    separate_trips_0_hlc = separateCardCombByCardNum(trips_with_0_heart_level_card, True)
    
    # 0 wild cards
    for i in range(13):
        t1 = separate_trips_0_hlc[i].copy()
        t2 = separate_trips_0_hlc[i + 1].copy()
        if not (len(t1) == 0 or len(t2) == 0):
            temp = list(product(t1, t2))
            for prod in temp:
                two_trip = prod[0].copy() + prod[1].copy()
                all_two_trips.append(two_trip)
    
    if card_dict[heart_level_card] >= 1:
        trips_with_1_heart_level_card = extractCardCombWith_N_HeartLevelCard(trips, heart_level_card, 1)
        separate_trips_1_hlc = separateCardCombByCardNum(trips_with_1_heart_level_card, True)
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
    
    
    if card_dict[heart_level_card] == 2:
        trips_with_2_heart_level_card = extractCardCombWith_N_HeartLevelCard(trips, heart_level_card, 2)
        separate_trips_2_hlc = separateCardCombByCardNum(trips_with_2_heart_level_card, True)
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

def findAllThreeWithTwos(card_dict : Dict[str, int], heart_level_card : str, pairs : List[List[str]], trips : List[List[str]]) -> List[List[str]]:
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
    
    pairs_with_0_heart_level_card = extractCardCombWith_N_HeartLevelCard(pairs, heart_level_card, 0)
    separate_pairs_0_hlc = separateCardCombByCardNum(pairs_with_0_heart_level_card)
    pairs_with_1_heart_level_card = None
    separate_pairs_1_hlc = None
    
    trips_with_0_heart_level_card = extractCardCombWith_N_HeartLevelCard(trips, heart_level_card, 0)
    separate_trips_0_hlc = separateCardCombByCardNum(trips_with_0_heart_level_card)
    trips_with_1_heart_level_card = None
    separate_trips_1_hlc = None
    trips_with_2_heart_level_card = None
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
    
    if card_dict[heart_level_card] >= 1:
        pairs_with_1_heart_level_card = extractCardCombWith_N_HeartLevelCard(pairs, heart_level_card, 1)
        separate_pairs_1_hlc = separateCardCombByCardNum(pairs_with_1_heart_level_card)
        trips_with_1_heart_level_card = extractCardCombWith_N_HeartLevelCard(trips, heart_level_card, 1)
        separate_trips_1_hlc = separateCardCombByCardNum(trips_with_1_heart_level_card)
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
    
    if card_dict[heart_level_card] == 2:
        trips_with_2_heart_level_card = extractCardCombWith_N_HeartLevelCard(trips, heart_level_card, 2)
        separate_trips_2_hlc = separateCardCombByCardNum(trips_with_2_heart_level_card)
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

def findAllBombs(card_dict : Dict[str, int], heart_level_card : str) -> List[List[str]]:
    allBombs = list()
    num_heart_level_card = card_dict[heart_level_card]
    bomb_size_upper = 8 + num_heart_level_card
    for s in range(4, bomb_size_upper + 1):
        bombs = findAllBombsWithSize(card_dict, s, heart_level_card)
        allBombs.extend(bombs)
    return allBombs

def findAllBombsWithSize(card_dict : Dict[str, int], size : int, heart_level_card : str) -> List[List[str]]:
    '''
    This function does not find JOKER Bomb.
    '''
    bombs = list()
    num_heart_level_card = card_dict[heart_level_card]
    
    for i in range(2, 15):
        raw_cards = extractCardWithCardNum(card_dict, str(i) if i < 10 else NumToCardNum[i])
        cards = [card for card in raw_cards if card != heart_level_card]
        if len(cards) < size - num_heart_level_card:
            continue
        if len(cards) == size - num_heart_level_card:
            extra = [heart_level_card, heart_level_card]
            if num_heart_level_card == 1:
                _ = extra.pop()
                assert len(extra) == 1
            bombs.append((cards + extra).copy())
        else:
            temp = list(combinations(cards, size))
            for b in temp:
                bombs.append(list(b))
            if num_heart_level_card >= 1:
                temp2 = list(combinations(cards, size - 1))
                for b2 in temp2:
                    bombs.append(list(b2) + [heart_level_card])
            if num_heart_level_card >= 2:
                temp3 = list(combinations(cards, size - 2))
                for b3 in temp3:
                    bombs.append(list(b3) + [heart_level_card, heart_level_card])
    
    return bombs

def findJokerBomb(card_dict : Dict[str, int]) -> List[List[str]]:
    answer = list()
    
    if card_dict['SB'] == 2 and card_dict['HR'] == 2:
        answer.append(['SB', 'SB', 'HR', 'HR'])

    return answer

def findAllStraightAndStraightFlush(card_dict: Dict[str, int], heart_level_card : str) -> List[List[str]]:
    allStraight = list()
    num_heart_level = card_dict[heart_level_card]
    for i in range(1, 11):
        straights = findAllStraightFrom(card_dict, heart_level_card, i, 4, num_heart_level)
        if len(straights) > 0:
            for straight in straights:
                allStraight.append(list(straight))
    return allStraight

def findAllStraightFrom(card_dict : Dict[str, int], heart_level_card : str, current : int, remain : int, num_heart_level_card : int) -> List[List[str]]:
    card_num = NumToCardNum[current] if current == 1 or current >= 10 else str(current)
    answer = list()
    use_heart_level_card = False
    for suit in SUITS:
        card = suit + card_num
        if card_dict[card] > 0:
            if remain == 0:
                if card == heart_level_card and num_heart_level_card > 0:
                    use_heart_level_card = True
                    answer.append([card])
                elif card != heart_level_card:
                    answer.append([card])
            else:
                next_straight = list()
                if card == heart_level_card and num_heart_level_card > 0:
                    use_heart_level_card = True
                    next_straight = findAllStraightFrom(card_dict, heart_level_card, current + 1, remain - 1, num_heart_level_card - 1)
                elif card != heart_level_card:
                    next_straight = findAllStraightFrom(card_dict, heart_level_card, current + 1, remain - 1, num_heart_level_card)
                if len(next_straight) > 0:
                    for data in next_straight:
                        t = [card] + list(data)
                        answer.append(t)
    if not use_heart_level_card and num_heart_level_card > 0:
        if remain == 0:
            answer.append([heart_level_card])
        else:
            next_straight = findAllStraightFrom(card_dict, heart_level_card, current + 1, remain - 1, num_heart_level_card - 1)
            if len(next_straight) > 0:
                for data in next_straight:
                    t = [heart_level_card] + list(data)
                    answer.append(t)
    return answer

def isStraightFlush(cardComb : List[str], heart_level_card : str) -> bool:
    '''
    @param cardComb
    This is the Straight Combination that we are going to test.
    We assume it is a legal Straight Combination.
    '''
    if len(cardComb) != 5:
        return False
    suit = 'None'
    for card in cardComb:
        if card != heart_level_card:
            if suit not in SUITS:
                suit = card[0]
            elif suit in SUITS and card[0] != suit:
                return False
    return True

def powerOfStraight(straight : List[str], heart_level_card : str) -> int:
    p = 'None'
    num = -1
    for i in range(3):
        if straight[i] != heart_level_card:
            num = CardNumToNum[straight[i][1]] % 13 if not straight[i][1].isnumeric() else int(straight[i][1])
            num -= i
    p = NumToCardNum[num] if num == 1 or num == 10 else str(num)
    return p

def updateCardCombsAfterAction(card_combs : List[List], card_dict : Dict[str, int], action : Optional[List]) -> Tuple[bool, List[List]]:
    if action == None:
        return True, card_combs.copy()
    
    contain_action = False
    
    for comb in card_combs:
        if isSameAction(comb, action):
            contain_action = True
    
    if not contain_action:
        return False, card_combs.copy()
    
    action_dict = cardsToDict(action[2], 0)
    _ = action_dict.pop('total')
    keys = list(action_dict.keys())

    for key in keys:
        card_dict[key] -= action_dict[key]
        card_dict['total'] -= action_dict[key]

    new_combs = list()
    
    for comb in card_combs:
        can_be_added = True
        for key in keys:
            if comb[2].count(key) == 2:
                can_be_added = False
            elif comb[2].count(key) == 1:
                if action_dict[key] == 2 or (action_dict[key] == 1 and card_dict[key] == 0):
                    can_be_added = False
            if not can_be_added:
                break
        if can_be_added:
            new_combs.append(comb)
    
    return True, new_combs

def isSameAction(action1 : List[str], action2 : List[str]) -> bool:
    if action1[0] != action2[0]:
        return False
    if action1[1] != action2[1]:
        return False
    if len(action1[2]) != len(action2[2]):
        return False
    for i in range(len(action1[2])):
        if action1[2][i] != action2[2][i]:
            return False
    return True

if __name__ == "__main__":
    card_lists = generateNRandomCardLists((12, ))
    card_dict = cardsToDict(card_lists[0])
    card_comb = findAllCombs(card_dict, 1)
    action = ['Single', '2', ['H2']]
    print(card_comb)
    print(card_dict)
    print(action)
    ans, card_comb = updateCardCombsAfterAction(card_comb, card_dict, action)
    print(ans)
    print(card_comb)
    print(card_dict)
    