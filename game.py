import random

class Card:
    SUITS = ['H', 'D', 'C', 'S']
    RANKS_1 = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    JOKERS = ['SB', 'HR']  # 小王和大王

    def __init__(self, suit=None, rank=None):
        if rank in self.JOKERS:
            self.suit = None
            self.rank = rank
        else:
            self.suit = suit
            self.rank = rank

    def __repr__(self):
        if self.rank in self.JOKERS:
            return self.rank
        return f"{self.suit}{self.rank}"

class Deck:
    def __init__(self):
        # 创建52张常规牌和2张Joker（两副牌）
        self.cards = [Card(suit, rank) for suit in Card.SUITS for rank in Card.RANKS_1] * 2
        self.cards += [Card(rank=joker) for joker in Card.JOKERS] * 2
        random.shuffle(self.cards)

    def deal_hands(self, num_cards_player1, num_cards_player2):
        
        hand1 = PokerHand([self.cards.pop() for _ in range(num_cards_player1)])
        hand2 = PokerHand([self.cards.pop() for _ in range(num_cards_player2)])
        return hand1, hand2

# 输出: Hand: H3, DK, SB
class PokerHand:
    def __init__(self, cards):
        self.cards = cards

    def __repr__(self):
        return f"Hand: {', '.join(map(str, self.cards))}"

class PokerGame:
    def __init__(self):
        self.deck = Deck()
        self.hands_player1 = []
        self.hands_player2 = []

    def deal_hands(self, num_cards_player1, num_cards_player2):
        hand1, hand2 = self.deck.deal_hands(num_cards_player1, num_cards_player2)
        self.hands_player1 = hand1.cards
        self.hands_player2 = hand2.cards

    def show_hands(self):
        hand1 = PokerHand(self.hands_player1)
        hand2 = PokerHand(self.hands_player2)
        print(f"Player 1: {hand1}")
        print(f"Player 2: {hand2}")


if __name__ == "__main__":
    game = PokerGame()
    game.deal_hands()
    game.show_hands()
