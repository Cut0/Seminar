import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# データファイルの存在するフォルダ
# ここでは python プログラムと同じフォルダに存在するものとする
DATA_DIR = './'

# 学習データセット（CSVファイル）のファイル名
TRAIN_DATA_FILE = os.path.join(DATA_DIR, './log/ai_player_log.csv')

# 学習データセットおよびテストセットのエンコード形式
# ここでは UTF-8 を想定する
FILE_ENCODING = 'utf_8'

# 学習条件の設定
BATCH_SIZE = 64  # バッチサイズ
N_EPOCHS = 100  # 何エポック分，学習処理を回すか

# 学習結果を保存するファイルのファイル名
MODEL_FILE = 'models/trained_model.pth'

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
        # 数負けとバースト負けの行は読み飛ばす
        if row[3] == "lose" or row[3] == "bust":
            continue

        # 空の入力ベクトル(5要素)を用意する
        vec = [0] * 5

        # 入力ベクトルに値を設定する
        vec[0] = float(row[0])
        vec[1] = float(row[4])
        vec[2] = float(row[5])
        vec[3] = float(row[6])
        vec[4] = float(row[7])

        # 出力側のデータ（正解ラベルのデータ）を作成する
        if row[1] == 'HIT':
            lab = 0
        elif row[1] == 'DOUBLE DOWN':
            lab = 1
        elif row[1] == 'SURRENDER':
            lab = 2
        elif row[1] == 'STAND':
            lab = 3
        else:
            continue

        x_set.append(vec)
        y_set.append(lab)

    # ファイルをクローズ
    f.close()

    # 読み込んだデータを numpy.ndarray 型に変換
    x_set = np.asarray(x_set, dtype=np.float32)  # 32bit浮動小数点数型に
    y_set = np.asarray(y_set, dtype=np.int64)  # 64bit整数型に

    return x_set, y_set

# ニューラルネットワーク


class myMLP(nn.Module):
    def __init__(self):
        super(myMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 4),
        )

    def forward(self, x):
        return self.layers(x)


# C言語のメイン関数に相当するもの
if __name__ == '__main__':
    # データ読み込み
    x_train, y_train = read_numerical_data(
        TRAIN_DATA_FILE, encoding=FILE_ENCODING)
    n_samples = len(x_train)  # 読み込んだデータの総数を変数 n_samples に記憶しておく

    # ニューラルネットワークの用意
    net = myMLP()

    # 損失関数の定義
    # ここでは Softmax + CrossEntropy損失 を用いる
    loss_func = nn.CrossEntropyLoss()

    # デバイスの指定とオプティマイザーの用意（基本このままでOK）
    device = 'cpu'
    net = net.to(device)
    optimizer = optim.Adam(net.parameters())

    # 学習データセットを用いてニューラルネットワークを学習
    for epoch in range(N_EPOCHS):

        print('Epoch {0}:'.format(epoch + 1))

        net.train()
        sum_loss = 0
        perm = np.random.permutation(n_samples)  # 学習データの使用順序をランダム化するために使用
        for i in range(0, n_samples, BATCH_SIZE):
            net.zero_grad()
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
