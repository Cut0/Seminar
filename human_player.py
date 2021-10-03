import socket
import tkinter as tk
from classes import Hand
from config import PORT, BET, INITIAL_MONEY, MAX_CARDS_PAR_GAME


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

### ここまで ###

LOG_FILE = './log/human_player_log.csv'


# ゲームを開始する
def game_start():

    global money, current_bet, player_hand, dealer_hand, soc

    # 前回ゲームの結果表示を削除
    result_text.set("")

    # ディーラープログラムに接続する
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ベット額を現在の所持金から引く
    current_bet = BET
    money -= current_bet
    money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, current_bet)) # 金額表示を更新

    # ディーラーからカード情報を受信
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    pc1, pc2, dc = msg.split(',')
    pc1 = int(pc1)
    pc2 = int(pc2)
    dc = int(dc)
    player_hand.clear()
    dealer_hand.clear()
    player_hand.append(pc1)
    player_hand.append(pc2)
    dealer_hand.append(dc)

    # 画面表示を更新（配られたカードを表示）
    player_score_text.set("(score: {0})".format(player_hand.get_score()))
    dealer_score_text.set("(score: {0})".format(dealer_hand.get_score()))
    pc_img = [tk.PhotoImage(file="./imgs/{0}.png".format(pc1+1)), tk.PhotoImage(file="./imgs/{0}.png".format(pc2+1))]
    dc_img = [tk.PhotoImage(file="./imgs/{0}.png".format(dc+1)), tk.PhotoImage(file="./imgs/ura.png")]
    empty_img = tk.PhotoImage(file="./imgs/0.png")
    for i in range(0, 2):
        player_canvas[i].photo = pc_img[i]
        dealer_canvas[i].photo = dc_img[i]
        player_canvas[i].itemconfig(player_canvas_img[i], image=player_canvas[i].photo)
        dealer_canvas[i].itemconfig(dealer_canvas_img[i], image=dealer_canvas[i].photo)
    for i in range(2, MAX_CARDS_PAR_GAME):
        player_canvas[i].photo = empty_img
        dealer_canvas[i].photo = empty_img
        player_canvas[i].itemconfig(player_canvas_img[i], image=player_canvas[i].photo)
        dealer_canvas[i].itemconfig(dealer_canvas_img[i], image=dealer_canvas[i].photo)

    # アクションボタンをアクティブにする
    ht_button['state'] = tk.NORMAL
    st_button['state'] = tk.NORMAL
    dd_button['state'] = tk.NORMAL
    sr_button['state'] = tk.NORMAL

    # スタートボタンと終了を非アクティブにする
    start_button['state'] = tk.DISABLED
    quit_button['state'] = tk.DISABLED


# ディーラーに HIT を要求する
def hit():

    global money, current_bet, player_hand, dealer_hand, soc

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # ディーラーにメッセージを送信
    soc.send(bytes("hit", 'utf-8'))

    # 配布されたカード，現在のスコア，バーストしたか否かをディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    pc, score, status, rate = msg.split(',')
    pc = int(pc)
    score = int(score)
    rate = float(rate)
    player_hand.append(pc)

    # 配布されたカードを画面に表示
    pc_img = tk.PhotoImage(file="./imgs/{0}.png".format(pc+1))
    n = len(player_hand.cards) - 1
    player_canvas[n].photo = pc_img
    player_canvas[n].itemconfig(player_canvas_img[n], image=player_canvas[n].photo)
    player_score_text.set("(score: {0})".format(player_hand.get_score()))

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    print('{0},HIT,{1},{2}'.format(prev_score, score, status), file=logf)

    # バーストした場合はゲーム終了
    if status == 'bust':

        # ディーラーとの通信をカット
        soc.close()

        # アクションボタンを非アクティブにする
        ht_button['state'] = tk.DISABLED
        st_button['state'] = tk.DISABLED
        dd_button['state'] = tk.DISABLED
        sr_button['state'] = tk.DISABLED

        # スタートボタンと終了ボタンをアクティブにする
        start_button['state'] = tk.NORMAL
        quit_button['state'] = tk.NORMAL

        # 金額表示を更新
        current_bet = 0
        money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, current_bet))

        # 勝敗表示を更新
        result_text.set("Bust...")
        result_label['fg'] = 'blue'


# ディーラーに STAND を要求する
def stand():

    global money, current_bet, player_hand, dealer_hand, soc

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # ディーラーにメッセージを送信
    soc.send(bytes("stand", 'utf-8'))

    # スコア，勝敗結果，配当倍率，ディーラーカードをディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    msg = msg.split(',')
    result = msg[1]
    score = int(msg[0])
    rate = float(msg[2])

    # ディーラーのカードを画面に表示
    for i in range(3, len(msg)):
        dc = int(msg[i])
        dealer_hand.append(dc) # ディーラーの手配を更新
        dc_img = tk.PhotoImage(file="./imgs/{0}.png".format(dc+1))
        dealer_canvas[i-2].photo = dc_img
        dealer_canvas[i-2].itemconfig(dealer_canvas_img[i-2], image=dealer_canvas[i-2].photo)
    player_score_text.set("(score: {0})".format(player_hand.get_score()))
    dealer_score_text.set("(score: {0})".format(dealer_hand.get_score()))

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    print('{0},STAND,{1},{2}'.format(prev_score, score, result), file=logf)

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # アクションボタンを非アクティブにする
    ht_button['state'] = tk.DISABLED
    st_button['state'] = tk.DISABLED
    dd_button['state'] = tk.DISABLED
    sr_button['state'] = tk.DISABLED

    # スタートボタンと終了ボタンをアクティブにする
    start_button['state'] = tk.NORMAL
    quit_button['state'] = tk.NORMAL

    # 所持金額を更新
    money += int(current_bet * rate)

    # 金額表示を更新
    current_bet = 0
    money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, current_bet))

    # 勝敗表示を更新
    if result == 'lose':
        result_text.set("Lose...")
        result_label['fg'] = 'blue'
    elif result == 'win':
        result_text.set("Win!!")
        result_label['fg'] = 'red'
    else:
        result_text.set("Draw")
        result_label['fg'] = 'green'


# ディーラーに DOUBLE DOWN を要求する
def double_down():

    global money, current_bet, player_hand, dealer_hand, soc

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # 今回のみベットを倍にする
    money -= current_bet
    current_bet *= 2
    money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, current_bet)) # 金額表示を更新

    # ディーラーにメッセージを送信
    soc.send(bytes("double_down", 'utf-8'))

    # 配布されたカード，スコア，勝敗結果，配当倍率，ディーラーカードディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    msg = msg.split(',')
    result = msg[2]
    pc = int(msg[0])
    score = int(msg[1])
    rate = float(msg[3])
    player_hand.append(pc)

    # 配布されたカードを画面に表示
    pc_img = tk.PhotoImage(file="./imgs/{0}.png".format(pc+1))
    n = len(player_hand.cards) - 1
    player_canvas[n].photo = pc_img
    player_canvas[n].itemconfig(player_canvas_img[n], image=player_canvas[n].photo)
    player_score_text.set("(score: {0})".format(player_hand.get_score()))

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    print('{0},DOUBLE DOWN,{1},{2}'.format(prev_score, score, result), file=logf)

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # アクションボタンを非アクティブにする
    ht_button['state'] = tk.DISABLED
    st_button['state'] = tk.DISABLED
    dd_button['state'] = tk.DISABLED
    sr_button['state'] = tk.DISABLED

    # スタートボタンと終了ボタンをアクティブにする
    start_button['state'] = tk.NORMAL
    quit_button['state'] = tk.NORMAL

    # バーストした場合
    if result == 'bust':

        # 金額表示を更新
        current_bet = 0
        money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, current_bet))

        # 勝敗表示を更新
        result_text.set("Bust...")
        result_label['fg'] = 'blue'

    # バーストしなかった場合
    else:

        # ディーラーのカードを画面に表示
        for i in range(4, len(msg)):
            dc = int(msg[i])
            dealer_hand.append(dc) # ディーラーの手配を更新
            dc_img = tk.PhotoImage(file="./imgs/{0}.png".format(dc+1))
            dealer_canvas[i-3].photo = dc_img
            dealer_canvas[i-3].itemconfig(dealer_canvas_img[i-3], image=dealer_canvas[i-3].photo)
        dealer_score_text.set("(score: {0})".format(dealer_hand.get_score()))

        # 所持金額を更新
        money += int(current_bet * rate)

        # 金額表示を更新
        current_bet = 0
        money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, current_bet))

        # 勝敗表示を更新
        if result == 'lose':
            result_text.set("Lose...")
            result_label['fg'] = 'blue'
        elif result == 'win':
            result_text.set("Win!!")
            result_label['fg'] = 'red'
        else:
            result_text.set("Draw")
            result_label['fg'] = 'green'


# ディーラーに SURRENDER を要求する
def surrender():

    global money, current_bet, player_hand, dealer_hand, soc

    # 行動を実行する前のスコアを求めておく
    prev_score = player_hand.get_score()

    # ディーラーにメッセージを送信
    soc.send(bytes("surrender", 'utf-8'))

    # スコア，サレンダー受付の返事，配当倍率をディーラーから通知してもらう
    msg = soc.recv(1024)
    msg = msg.decode("utf-8")
    score, status, rate = msg.split(',')
    score = int(score)
    rate = float(rate)

    # 行動前スコア，行動，行動後スコア，行動後ステータスをログに記録しておく
    print('{0},SURRENDER,{1},{2}'.format(prev_score, score, status), file=logf)

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # アクションボタンを非アクティブにする
    ht_button['state'] = tk.DISABLED
    st_button['state'] = tk.DISABLED
    dd_button['state'] = tk.DISABLED
    sr_button['state'] = tk.DISABLED

    # スタートボタンと終了ボタンをアクティブにする
    start_button['state'] = tk.NORMAL
    quit_button['state'] = tk.NORMAL

    # 所持金額を更新
    money += int(current_bet * rate)

    # 金額表示を更新
    current_bet = 0
    money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, current_bet))

    # 勝敗表示を更新
    result_text.set("Surrendered")
    result_label['fg'] = 'green'


# ゲームを終了する
def game_quit():

    root.destroy()


# メインウィンドウの作成
root = tk.Tk()
root.title('Black Jack')
root.geometry('{0}x440'.format(200+120*MAX_CARDS_PAR_GAME))
root.protocol('WM_DELETE_WINDOW', (lambda: 'pass')())

# 残額表示オブジェクトの作成と設置
money_text = tk.StringVar()
money_text.set("money: {0:5}$".format(money))
money_label = tk.Label(root, textvariable=money_text, font=('Arial', '12', 'bold'))
money_label.place(x=-20+120*MAX_CARDS_PAR_GAME, y=10)

# 勝敗表示オブジェクトの作成と設置
result_text = tk.StringVar()
result_text.set("")
result_label = tk.Label(root, textvariable=result_text, font=('Arial', '24', 'bold'))
result_label.place(x=330, y=210)

# スコア表示用オブジェクトの作成と設置
player_score_text = tk.StringVar()
dealer_score_text = tk.StringVar()
player_score_text.set("(score:   )")
dealer_score_text.set("(score:   )")
player_score_label = tk.Label(root, textvariable=player_score_text, font=('Arial', '14', 'bold'))
dealer_score_label = tk.Label(root, textvariable=dealer_score_text, font=('Arial', '14', 'bold'))
player_score_label.place(x=160, y=240)
dealer_score_label.place(x=160, y=10)

# カード表示用キャンバスの作成と設置
empty_img = tk.PhotoImage(file="./imgs/0.png")
dealer_label = tk.Label(text="Dealer's cards", font=('Arial', '14', 'bold'))
player_label = tk.Label(text="Player's cards", font=('Arial', '14', 'bold'))
dealer_label.place(x=10, y=10)
player_label.place(x=10, y=240)
dealer_canvas = [0] * MAX_CARDS_PAR_GAME
player_canvas = [0] * MAX_CARDS_PAR_GAME
dealer_canvas_img = [0] * MAX_CARDS_PAR_GAME
player_canvas_img = [0] * MAX_CARDS_PAR_GAME
for i in range(0, MAX_CARDS_PAR_GAME):
    dealer_canvas[i] = tk.Canvas(width=112, height=160)
    dealer_canvas[i].photo = empty_img
    dealer_canvas_img[i] = dealer_canvas[i].create_image(0, 0, image=dealer_canvas[i].photo, anchor=tk.NW)
    dealer_canvas[i].place(x=10+120*i, y=40)
    player_canvas[i] = tk.Canvas(width=112, height=160)
    player_canvas[i].photo = empty_img
    player_canvas_img[i] = player_canvas[i].create_image(0, 0, image=player_canvas[i].photo, anchor=tk.NW)
    player_canvas[i].place(x=10+120*i, y=270)

# ボタンの作成と設置
action_label = tk.Label(text="Action:", font=('Arial', '14', 'bold'))
action_label.place(x=40+120*MAX_CARDS_PAR_GAME, y=160)
start_button = tk.Button(width=14, text='Game Start', font=('Arial', '12', 'bold'), command=game_start)
ht_button = tk.Button(width=15, text='HIT', font=('Arial', '12'), state=tk.DISABLED, command=hit)
st_button = tk.Button(width=15, text='STAND', font=('Arial', '12'), state=tk.DISABLED, command=stand)
dd_button = tk.Button(width=15, text='DOUBLE DOWN', font=('Arial', '12'), state=tk.DISABLED, command=double_down)
sr_button = tk.Button(width=15, text='SURRENDER', font=('Arial', '12'), state=tk.DISABLED, command=surrender)
quit_button = tk.Button(width=14, text='Quit', font=('Arial', '12', 'bold'), command=game_quit)
start_button.place(x=40+120*MAX_CARDS_PAR_GAME, y=50)
ht_button.place(x=40+120*MAX_CARDS_PAR_GAME, y=190)
st_button.place(x=40+120*MAX_CARDS_PAR_GAME, y=225)
dd_button.place(x=40+120*MAX_CARDS_PAR_GAME, y=260)
sr_button.place(x=40+120*MAX_CARDS_PAR_GAME, y=295)
quit_button.place(x=40+120*MAX_CARDS_PAR_GAME, y=400)

# ログファイルを開く
logf = open(LOG_FILE, 'a')

# メインループ開始
root.mainloop()

# ログファイルを閉じる
logf.close()
