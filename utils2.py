from typing import List, Dict

import tensorflow as tf

from utils import NumToCardNum

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