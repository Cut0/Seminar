import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# train.py で定義した関数　read_data およびニューラルネットワーク myMLP をインポートする
# train.py で「myMLP」の部分の名称を変えている場合は，下の行の「myMLP」も同様に変更する必要がある
from train import read_numerical_data, read_categorical_data, myMLP

# tran.py で指定した定数値（フォルダ名やファイル名）も一応インポートし，
# 本プログラムでも使えるようにしておく
# ここは基本的に変更しなくてOK
from train import DATA_DIR, FILE_ENCODING, BATCH_SIZE, MODEL_FILE


# テストデータセット（CSVファイル）のファイル名
# ここでは例として mobile_phone_test.csv を用い，
# 携帯端末のスペックから価格帯（0～3の4段階）を分類する分類器をテストする場合を例にとる
# mushroom_test.csv を用い，食べられるキノコか毒キノコかを分類する場合も試してみましょう．
TEST_DATA_FILE = os.path.join(DATA_DIR, 'mobile_phone_test.csv')


# ニューラルネットワーク net にデータ vec を入力して分類結果を得る関数
def predict(net, vec):

    net.eval() # 必須．現状では「とりあえず記載しておく」という理解でOK

    x = torch.tensor([vec], device=device) # 入力ベクトル vec をPyTorch用の型に変換
    z = F.softmax(net(x), dim=1) # AIによる推定．Softmaxを適用して確率（っぽい値）を返すようにする
    z = z.to('cpu').detach().numpy().copy()[0] # 結果をnumpy.ndarray 型に変換して返却

    return z


# C言語のメイン関数に相当するもの
if __name__ == '__main__':

    # データ読み込み
    # 数値データの場合は read_numerical_data, カテゴリデータの場合は read_categorical_data を使う
    x_test, y_test = read_numerical_data(TEST_DATA_FILE, encoding=FILE_ENCODING)
    n_samples = len(x_test) # 読み込んだデータの総数を変数 n_samples に記憶しておく

    # ニューラルネットワークの用意
    net = myMLP() # 上で指定した名前（「myMLP」の部分）を指定する
    net.load_state_dict(torch.load(MODEL_FILE)) # 学習済みのパラメータをファイルからロードする

    # デバイスの指定とオプティマイザーの用意（基本このままでOK）
    device = 'cpu'
    net = net.to(device)

    # テストデータを入力して分類精度を評価
    net.eval() # 必須．現状では「とりあえず記載しておく」という理解でOK
    n_misestimated_samples = 0
    for i in range(0, n_samples):
        output = predict(net, x_test[i]) # i番目のテストデータ x_test[i] をニューラルネットワーク net に入力
        z = np.argmax(output) # ニューラルネットワークの出力に基づいて分類結果を決定（何番目の次元が最も大きい値を持つかを求める）
        print(output) # 確認用出力（ニューラルネットワークの生の出力値）
        print(z) # 確認用出力（最も値の大きい次元の番号，これを正解値 y_test[i] と比較して精度評価）
        if z != y_test[i]:
            n_misestimated_samples += 1 # 正解と推定値で値が異なるデータの数を数える
    acc = (n_samples - n_misestimated_samples) / n_samples # 分類精度を求める
    print('accuracy = {0:.2f}%'.format(100 * acc)) # 分類精度をパーセント表現で表示
    print('')
