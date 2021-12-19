import socket
from classes import CardSet, Hand
from config import PORT, N_DECKS, N_GAMES, MAX_CARDS_PAR_GAME


# 手札の初期化
def init_hands(card_set, dealer_hand, player_hand):

    # まず，手配を空にする
    dealer_hand.clear()
    player_hand.clear()

    # 2枚ずつドロー
    dealer_hand.append(card_set.draw())
    dealer_hand.append(card_set.draw())
    player_hand.append(card_set.draw())
    player_hand.append(card_set.draw())


# 勝敗判定
def judge(dealer_hand, player_hand):

    if player_hand.is_busted():
        return "lose", 0
    elif player_hand.is_nbj():
        if dealer_hand.is_nbj():
            return "draw", 1
        else:
            return "win", 2.5
    else:
        if dealer_hand.is_busted():
            return "win", 2
        elif dealer_hand.is_nbj():
            return "lose", 0
        else:
            player_score = player_hand.get_score()
            dealer_score = dealer_hand.get_score()
            if player_score > dealer_score:
                return "win", 2
            elif player_score < dealer_score:
                return "lose", 0
            else:
                return "draw", 1


# カードIDからスートと数字の情報のみを抽出する
def get_info(card):
    return card % 52


# ここから処理開始
if __name__ == "__main__":

    # カードセットを準備
    card_set = CardSet(n_decks=N_DECKS)

    # 通信用ソケットを作成
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.settimeout(1.0)
    soc.bind((socket.gethostname(), PORT))
    soc.listen(1)
    print("The dealer program has started!!")
    print()
    print("Wainting for a new player ...")

    # 現在が何回目のゲームかを示す変数を用意し，0で初期化
    game_ID = 0

    # Ctrl+C で停止されるまで，無限ループでゲームを続ける
    while True:

        try:
            # プレイヤーからの通信待ち状態に入る
            player_soc, address = soc.accept()
        except socket.timeout:
            pass
        except:
            raise
        else:

            print("A player has come.")

            if game_ID % N_GAMES == 0:
                card_set.shuffle()  # N_GAMES 回ゲームを行ったらカードセットをシャッフル
                print("Card set has been shuffled.")
            game_ID += 1
            print("Num. remaining cards: ", card_set.remaining_cards())

            # プレイヤーからの通信をキャッチしたら，まずディーラーとプレイヤーに2枚ずつカードを配る
            print("Game start!!")
            dealer_hand = Hand()
            player_hand = Hand()
            init_hands(card_set, dealer_hand, player_hand)

            # プレイヤーカード2枚とディーラーカード1枚をプレイヤーに開示
            # プレイヤーカード1，プレイヤーカード2，ディーラーカードの順で送信
            pc0 = get_info(player_hand.cards[0])
            pc1 = get_info(player_hand.cards[1])
            dc0 = get_info(dealer_hand.cards[0])
            player_soc.send(
                bytes("{0},{1},{2}".format(pc0, pc1, dc0), "utf-8"))

            # プレイヤーのアクションを受信して応答する（ループ処理）
            while True:

                # プレイヤーからメッセージを受信
                msg = player_soc.recv(1024)
                msg = msg.decode("utf-8")
                print("The player's action: ", msg)

                # HIT の場合
                if msg == "hit":

                    # プレイヤーに1枚カードを配る
                    player_hand.append(card_set.draw())

                    # 配布されたカード，現在のスコア，現在のステータス（バーストしたか否か），配当倍率（常に0，ダミー値）の
                    # 4つの情報をこの順でプレイヤーに通知
                    pc = get_info(player_hand.cards[-1])
                    player_score = player_hand.get_score()
                    player_status = "bust" if player_hand.is_busted() else "unsettled"
                    rate = 0
                    player_soc.send(bytes("{0},{1},{2},{3}".format(
                        pc, player_score, player_status, rate), "utf-8"))

                    print("The player's status: ", player_status)

                    # プレイヤーがバーストした場合はゲーム終了
                    if player_status == "bust":
                        break

                # STAND の場合
                elif msg == "stand":

                    # ルールに従ってディーラーにカードを追加
                    while dealer_hand.get_score() < 17 and len(dealer_hand.cards) < MAX_CARDS_PAR_GAME:
                        dealer_hand.append(card_set.draw())

                    # 勝敗を判定
                    player_status, rate = judge(dealer_hand, player_hand)

                    # 現在のスコア，現在のステータス（勝敗結果），配当倍率，ディーラーカード（最初の1枚以外）の
                    # 4つの情報をこの順でプレイヤーに通知
                    player_score = player_hand.get_score()
                    msg = "{0},{1},{2}".format(
                        player_score, player_status, rate)
                    for i in range(1, len(dealer_hand.cards)):
                        dc = get_info(dealer_hand.cards[i])
                        msg += ",{0}".format(dc)
                    player_soc.send(bytes(msg, "utf-8"))

                    print("The player' status: ", player_status)

                    # 結果によらずゲーム終了
                    break

                # DOUBLE DOWNの場合
                elif msg == "double_down":

                    # プレイヤーに1枚カードを配る
                    player_hand.append(card_set.draw())
                    player_score = player_hand.get_score()
                    pc = get_info(player_hand.cards[-1])

                    # プレイヤーがバーストした場合
                    if player_hand.is_busted():

                        # 配布されたカード，現在のスコア，現在のステータス（勝敗結果），配当倍率の
                        # 4つの情報をこの順でプレイヤーに通知
                        player_status = "bust"
                        rate = 0
                        player_soc.send(bytes("{0},{1},{2},{3}".format(
                            pc, player_score, player_status, rate), "utf-8"))

                    # プレイヤーがバーストしなかった場合
                    else:

                        # ルールに従ってディーラーにカードを追加
                        while dealer_hand.get_score() < 17 and len(dealer_hand.cards) < MAX_CARDS_PAR_GAME:
                            dealer_hand.append(card_set.draw())

                        # 勝敗を判定
                        player_status, rate = judge(dealer_hand, player_hand)

                        # 配布されたカード，現在のスコア，現在のステータス（勝敗結果），配当倍率，ディーラーカード（最初の1枚以外）の
                        # 5つの情報をこの順でプレイヤーに通知
                        msg = "{0},{1},{2},{3}".format(
                            pc, player_score, player_status, rate)
                        for i in range(1, len(dealer_hand.cards)):
                            dc = get_info(dealer_hand.cards[i])
                            msg += ",{0}".format(dc)
                        player_soc.send(bytes(msg, "utf-8"))

                    print("The player's status: ", player_status)

                    # 結果によらずゲーム終了
                    break

                # SURRENDERの場合
                elif msg == "surrender":

                    # 現在のスコア，現在のステータス（サレンダーを受け付けたこと），配当倍率の
                    # 3つの情報をこの順でプレイヤーに通知
                    player_score = player_hand.get_score()
                    player_status = "surrendered"
                    rate = 0.5
                    player_soc.send(bytes("{0},{1},{2}".format(
                        player_score, player_status, rate), "utf-8"))

                    print("The player's status: ", player_status)

                    # ゲーム終了
                    break

                # 定義されていないアクションは終了要求とみなす
                else:
                    break

            # 通信終了
            player_soc.close()
            print("The game has finished!")
            print()
            print("Wainting for a new player ...")

    soc.close()
