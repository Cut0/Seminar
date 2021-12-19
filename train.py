import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# データファイルの存在するフォルダ
# ここでは python プログラムと同じフォルダに存在するものとする
DATA_DIR = './'

# 学習データセット（CSVファイル）のファイル名
# ここでは例として mobile_phone_train.csv を用い，
# 携帯端末のスペックから価格帯（0～3の4段階）を分類する分類器を学習する場合を例にとる
# mushroom_train.csv を用い，食べられるキノコか毒キノコかを分類する場合も試してみましょう．
TRAIN_DATA_FILE = os.path.join(DATA_DIR, './log/ai_player_log.csv')

# 学習データセットおよびテストセットのエンコード形式
# ここでは UTF-8 を想定する
# ※ 日本語を含まないCSVファイルを扱うときは FILE_ENCODING = None とするのが良いです．
# ※ Windows で日本語を含むCSVファイルを扱うときは FILE_ENCODING = 'shift_jis' とすると正常に動作するかもしれません．
FILE_ENCODING = 'utf_8'


# 学習条件の設定（値を変更してみましょう）
BATCH_SIZE = 64  # バッチサイズ
N_EPOCHS = 100  # 何エポック分，学習処理を回すか

# 学習結果を保存するファイルのファイル名
MODEL_FILE = 'trained_model.pth'


# データ読み込み関数（数値データの場合）
# 例えば mobile_phone_train.csv を用いる場合はこちらを使う
def read_numerical_data(filename, encoding):

    # ファイル名 filename のファイルを CSV ファイルとしてオープン
    f = open(filename, 'r', encoding=encoding)
    reader = csv.reader(f)

    # ヘッダ（項目名が記載されている先頭の行）は読み捨てる
    next(reader)

    # データを読み込む
    x_set = []
    y_set = []
    for row in reader:  # 行ごとに処理を行う

        ''' 以下の部分を変更して入力ベクトルや出力データを変更してみましょう： ここから '''

        # まず，空の入力ベクトルを作成する
        # 例えば4要素のベクトル（4次元ベクトル）にするなら，次のように記載する
        vec = [0] * 7

        # 次に，入力ベクトルに値を設定する
        # 例えば，最初の次元に「2:処理速度」，次に「4:正面カメラのメガピクセル数」，
        # 以下「7:本体の厚さ」「8:本体の重量」とセットするなら，次のように記載する
        vec[0] = float(row[0])
        vec[1] = float(row[2])
        if row[3] == "win":
            vec[2] = 0
        elif row[3] == "lose":
            vec[2] = 1
        elif row[3] == "unsettled":
            vec[2] = 2
        elif row[3] == "surrendered":
            vec[2] = 3
        elif row[3] == "bust":
            vec[2] = 4
        else:
            vec[2] = 5
        vec[3] = float(row[4])
        vec[4] = float(row[5])
        vec[5] = float(row[6])
        vec[6] = float(row[7])

        # その後，出力側のデータ（正解ラベルのデータ）を作成する
        # 今回の例では，「20:価格帯」が正解ラベルになるので，次のように記述する
        if row[1] == 'HIT':
            lab = 0  # 正解ラベルが「0」（文字列としてのゼロ）のとき，そのラベルを表す整数値として 0 を設定
        elif row[1] == 'DOUBLE DOWN':
            lab = 1  # 正解ラベルが「1」（文字列としてのイチ）のとき，そのラベルを表す整数値として 1 を設定
        elif row[1] == 'HIT':
            lab = 2
        elif row[1] == 'STAND':
            lab = 3
        else:
            # 万が一，0～3のどれでもないものがあった場合は，その行自体を無視する
            continue

        ''' ここまで '''

        x_set.append(vec)
        y_set.append(lab)

    # ファイルをクローズ
    f.close()

    # 読み込んだデータを numpy.ndarray 型に変換
    x_set = np.asarray(x_set, dtype=np.float32)  # 32bit浮動小数点数型に
    y_set = np.asarray(y_set, dtype=np.int64)  # 64bit整数型に

    return x_set, y_set

# ニューラルネットワーク


class myMLP(nn.Module):  # 適当な名前（「myMLP」の部分 ）を設定する

    def __init__(self):
        super(myMLP, self).__init__()  # 「myMLP」の部分を，3行上で設定した名前と同じにする

        ''' 以下の部分を変更してニューラルネットワークの構造を変更してみましょう： ここから '''
        ''' 入力・出力の次元数は対象とするデータセット（今回なら携帯端末かキノコか）によって変わることに注意 '''

        self.layers = nn.Sequential(
            nn.Linear(7, 10),  # 入力層（入力は4次元ベクトル）→ 中間層一層目（パーセプトロン数10）
            nn.ReLU(),  # 活性化関数
            nn.Linear(10, 10),  # 中間層一層目（パーセプトロン数10）→ 中間層二層目（パーセプトロン数10）
            nn.Sigmoid(),  # 活性化関数
            # 中間層二層目（パーセプトロン数10）→ 出力層（今回の例では正解ラベルは0～3の4種類なので，出力は4次元ベクトル）
            nn.Linear(10, 4),
        )

        ''' ここまで '''

    def forward(self, x):
        return self.layers(x)


# C言語のメイン関数に相当するもの
if __name__ == '__main__':

    # データ読み込み
    # 数値データの場合は read_numerical_data, カテゴリデータの場合は read_categorical_data を使う
    x_train, y_train = read_numerical_data(
        TRAIN_DATA_FILE, encoding=FILE_ENCODING)
    n_samples = len(x_train)  # 読み込んだデータの総数を変数 n_samples に記憶しておく

    # ニューラルネットワークの用意
    net = myMLP()  # 上で指定した名前（「myMLP」の部分）を指定する

    # 損失関数の定義
    # ここでは Softmax + CrossEntropy損失 を用いる
    # 分類問題ではこれを用いるのが普通なので，変更の必要は基本的にないはず
    loss_func = nn.CrossEntropyLoss()

    # デバイスの指定とオプティマイザーの用意（基本このままでOK）
    device = 'cpu'
    net = net.to(device)
    optimizer = optim.Adam(net.parameters())

    # 学習データセットを用いてニューラルネットワークを学習
    for epoch in range(N_EPOCHS):

        print('Epoch {0}:'.format(epoch + 1))

        net.train()  # 必須．現状では「とりあえず記載しておく」という理解でOK
        sum_loss = 0
        perm = np.random.permutation(n_samples)  # 学習データの使用順序をランダム化するために使用
        for i in range(0, n_samples, BATCH_SIZE):
            net.zero_grad()  # 必須．現状では「とりあえず記載しておく」という理解でOK
            # 入力ベクトル（PyTorch用の型に変換して使用）
            x = torch.tensor(x_train[perm[i: i + BATCH_SIZE]], device=device)
            # 正解ラベル（PyTorch用の型に変換して使用）
            y = torch.tensor(y_train[perm[i: i + BATCH_SIZE]], device=device)
            loss = loss_func(net(x), y)  # 損失関数の値を計算
            loss.backward()  # 損失関数の微分を計算
            optimizer.step()  # 微分係数に従ってパラメータの値を更新
            sum_loss += float(loss) * len(x)
        sum_loss /= n_samples

        print('  train loss = {0:.6f}'.format(sum_loss))  # 損失関数の現在値を表示
        print('')

    # 学習結果をファイルに保存する
    net = net.to('cpu')
    torch.save(net.state_dict(), MODEL_FILE)
