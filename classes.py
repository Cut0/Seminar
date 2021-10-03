import numpy as np


# カードセット
class CardSet:

    def __init__(self, n_decks):
        self.n_cards = 52 * n_decks
        self.all_cards = np.arange(0, self.n_cards)
        self.pos = -1
        self.shuffle()

    # シャッフル
    def shuffle(self):
        np.random.shuffle(self.all_cards)
        self.pos = 0

    # 1枚ドロー
    def draw(self):
        card = self.all_cards[self.pos]
        self.pos += 1
        return card

    # 残りカード枚数を取得
    def remaining_cards(self):
        return self.n_cards - self.pos


# 手札
class Hand:

    def __init__(self):
        self.clear()

    # カード c を追加
    def append(self, c):
        self.cards.append(c)

    # 手札をクリア（0枚にする）
    def clear(self):
        self.cards = []

    # 現在のスコアを計算
    def get_score(self):
        tmp = []
        have_ace = False
        for i in self.cards:
            j = min(10, (i % 13) + 1)
            if j != 1:
                tmp.append(j)
            else:
                if have_ace:
                    tmp.append(1)
                else:
                    have_ace = True
        score = sum(tmp)
        if have_ace:
            if score + 11 > 21:
                score += 1
            else:
                score += 11
        return score

    # ナチュラルブラックジャックか否かを判定
    def is_nbj(self):
        if self.get_score() == 21 and len(self.cards) == 2:
            return True
        else:
            return False

    # バーストか否かを判定
    def is_busted(self):
        if self.get_score() > 21:
            return True
        else:
            return False
