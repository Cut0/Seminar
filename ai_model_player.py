import sys
import socket
import numpy as np
from classes import Hand
from config import PORT, BET, INITIAL_MONEY, N_GAMES
from predictor import output

### グローバル変数 ###

# 所持金の設定
money = INITIAL_MONEY

# 現在のベット額
current_bet = 0

# プレイヤーの手配
player_hand = Hand()

# ディーラーの手配
dealer_hand = Hand()

# 通信用ソケットとして用いる変数を準備しておく
soc = None

# 他にも自分で色々な変数を定義して良い
player_score = -1
dealer_card_number = -1
n_used_cards = 0
card_count = 0
game_number = 0
### ここまで ###

LOG_FILE = "./log/ai_model_player_log.csv"


def get_card_info(card):

    n = (card % 13) + 1
    if n == 1:
        num = "A"
    elif n == 11:
        num = "J"
    elif n == 12:
        num = "Q"
    elif n == 13:
        num = "K"
    else:
        num = "{0}".format(n)

    s = card // 13
    if s == 0:
        suit = "Spade"
    elif s == 1:
        suit = "Club"
    elif s == 2:
        suit = "Diamond"
    else:
        suit = "Heart"

    return suit + "-" + num


# ゲームを開始する
def game_start(n=1):

    global money, current_bet, player_hand, dealer_hand, soc

    print("Game {0} start.".format(n))
    print("  money: ", money, "$")

    # ディーラープログラムに接続する
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ベット額を現在の所持金から引く
    current_bet = BET
    money -= current_bet

    print("Action: BET")
    print("  money: ", money, "$")
    print("  bet: ", current_bet, "$")

    # ディーラーからカード情報を受信
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    pc1, pc2, dc = msg.split(",")
    pc1 = int(pc1)
    pc2 = int(pc2)
    dc = int(dc)
    player_hand.clear()
    dealer_hand.clear()
    player_hand.append(pc1)
    player_hand.append(pc2)
    dealer_hand.append(dc)

    print("Delaer gave cards.")
    print("  dealer-card: ", get_card_info(dc))
    print("  player-card 1: ", get_card_info(pc1))
    print("  player-card 2: ", get_card_info(pc2))
    print("  current score: ", player_hand.get_score())


# ディーラーに HIT を要求する
def hit():

    global money, current_bet, player_hand, dealer_hand, soc

    print("Action: HIT")

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # ディーラーにメッセージを送信
    soc.send(bytes("hit", "utf-8"))

    # 配布されたカード，現在のスコア，バーストしたか否かをディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    pc, score, status, rate = msg.split(",")
    pc = int(pc)
    score = int(score)
    rate = float(rate)
    player_hand.append(pc)

    print(
        "  player-card {0}: ".format(len(player_hand.cards)), get_card_info(pc))
    print("  current score: ", player_hand.get_score())

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    global n_used_cards
    global player_score
    global dealer_card_number

    print("{0},HIT,{1},{2},{3},{4},{5},{6}".format(prev_score, score,
                                                   status, n_used_cards, dealer_card_number, (player_hand.cards[0] % 13) + 1, (player_hand.cards[1] % 13) + 1), file=logf)

    # バーストした場合はゲーム終了
    if status == "bust":

        # ディーラーとの通信をカット
        soc.close()

        # 所持金額を更新
        current_bet = 0

        print("Game finished.")
        print("  result: bust")
        print("  money: ", money, "$")
        return True

    else:
        return False


# ディーラーに STAND を要求する
def stand():

    global money, current_bet, player_hand, dealer_hand, soc

    print("Action: STAND")
    print("  current score: ", player_hand.get_score())

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # ディーラーにメッセージを送信
    soc.send(bytes("stand", "utf-8"))

    # スコア，勝敗結果，配当倍率，ディーラーカードをディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    msg = msg.split(",")
    result = msg[1]
    score = int(msg[0])
    rate = float(msg[2])
    for i in range(3, len(msg)):
        dc = int(msg[i])
        dealer_hand.append(dc)
        print("  dealer-card {0}: ".format(i-1), get_card_info(dc))
    print("  dealer's score: ", dealer_hand.get_score())

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    global n_used_cards
    global player_score
    global dealer_card_number

    print("{0},STAND,{1},{2},{3},{4},{5},{6}".format(prev_score, score,
                                                     result, n_used_cards, dealer_card_number, (player_hand.cards[0] % 13) + 1, (player_hand.cards[1] % 13) + 1), file=logf)

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    money += int(current_bet * rate)
    current_bet = 0

    print("Game finished.")
    print("  result: ", result)
    print("  money: ", money, "$")
    return True


# ディーラーに DOUBLE DOWN を要求する
def double_down():

    global money, current_bet, player_hand, dealer_hand, soc

    print("Action: DOUBLE DOWN")

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # 今回のみベットを倍にする
    money -= current_bet
    current_bet *= 2

    print("  money: ", money, "$")
    print("  bet: ", current_bet, "$")

    # ディーラーにメッセージを送信
    soc.send(bytes("double_down", "utf-8"))

    # 配布されたカード，スコア，勝敗結果，配当倍率，ディーラーカードディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    msg = msg.split(",")
    result = msg[2]
    pc = int(msg[0])
    score = int(msg[1])
    rate = float(msg[3])
    player_hand.append(pc)

    print(
        "  player-card {0}: ".format(len(player_hand.cards)), get_card_info(pc))
    print("  current score: ", player_hand.get_score())

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    global n_used_cards
    global player_score
    global dealer_card_number

    print("{0},DOUBLE DOWN,{1},{2},{3},{4},{5},{6}".format(prev_score, score,
                                                           result, n_used_cards, dealer_card_number, (player_hand.cards[0] % 13) + 1, (player_hand.cards[1] % 13) + 1), file=logf)
    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # バーストした場合
    if result == "bust":

        # 所持金額を更新
        current_bet = 0

    # バーストしなかった場合
    else:

        for i in range(4, len(msg)):
            dc = int(msg[i])
            dealer_hand.append(dc)  # ディーラーの手配を更新
            print("  dealer-card {0}: ".format(i-2), get_card_info(dc))
        print("  dealer's score: ", dealer_hand.get_score())

        # 所持金額を更新
        money += int(current_bet * rate)
        current_bet = 0

    print("Game finished.")
    print("  result: ", result)
    print("  money: ", money, "$")
    return True


# ディーラーに SURRENDER を要求する
def surrender():

    global money, current_bet, player_hand, dealer_hand, soc

    print("Action: SURRENDER")

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # ディーラーにメッセージを送信
    soc.send(bytes("surrender", "utf-8"))

    # スコア，サレンダー受付の返事，配当倍率をディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    score, status, rate = msg.split(",")
    score = int(score)
    rate = float(rate)

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    global n_used_cards
    global player_score
    global dealer_card_number
    print("{0},SURRENDER,{1},{2},{3},{4},{5},{6}".format(prev_score, score,
                                                         status, n_used_cards, dealer_card_number, (player_hand.cards[0] % 13) + 1, (player_hand.cards[1] % 13) + 1), file=logf)
    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    money += int(current_bet * rate)
    current_bet = 0

    print("Game finished.")
    print("  result: ", status)
    print("  money: ", money, "$")
    return True


def model_strategy():
    # グローバル変数
    # 自分で追加定義したグローバル変数がある場合は，その変数名を下の行に追加すると関数内で使えるようになる
    global player_hand, dealer_hand
    global n_used_cards
    global player_score
    global dealer_card_number
    global card_count 
    global game_number

    # 「現在の状態」に保存
    player_score = player_hand.get_score()
    dealer_card_number = dealer_hand.cards[0] % 13 + 1

    # 自分のカードの数字を保存
    player_cards_number = []
    for card in player_hand.cards:
        player_cards_number.append(card % 13 + 1)

    have_ace = player_hand.have_ace
    ps = player_hand.get_score()

    print("プレイヤーのカード枚数", len(player_hand.cards))
    print("現在のプレイヤーのスコア", ps)

    # カードカウンティング
    def calculate_card_count(card_number):
        global card_count 
        if card_number >= 10 or card_number == 1:
            card_count -= 1
        elif card_number <= 6:
            card_count += 1

    if game_number % N_GAMES == 1: #シャッフルされるごとにcard_countを初期化
        card_count = 0

    for card in player_hand.cards:
        calculate_card_count(card % 13 + 1)
    for card in dealer_hand.cards:
        calculate_card_count(card % 13 + 1)   

    # n_used_cardsは使わない
    data = np.asarray([ps, dealer_card_number,
                       player_cards_number[0], player_cards_number[1], card_count], dtype=np.float32)

    ans = output(data)

    return_value = None  # 行う行動を保存
    if ans == "H":
        return_value = hit()
    elif ans == "S":
        return_value = stand()
    elif ans == "D":
        return_value = double_down()
    elif ans == "SR":
        return_value = surrender()

    if return_value is True:
        n = len(dealer_hand.cards)
        if n <= 1:
            n += 1
        n += len(player_hand.cards)
        n_used_cards += n
        print("今ゲームで使用したカードの枚数", n)
        print("これまでの全ゲームで使用したカードの枚数", n_used_cards)


    return return_value


# ゲーム実行
if __name__ == "__main__":

    n_games = 1 if len(sys.argv) < 2 else int(sys.argv[1])

    # ログファイルを開く
    logf = open(LOG_FILE, "a")

    for n in range(n_games):
        game_start(n+1)
        game_number += 1
        while True:
            if model_strategy():
                break
        print("")

    # ログファイルを閉じる
    logf.close()
