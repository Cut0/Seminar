import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# train.py で定義した関数　read_data およびニューラルネットワーク myMLP をインポートする
# train.py で「myMLP」の部分の名称を変えている場合は，下の行の「myMLP」も同様に変更する必要がある
from train import myMLP

# tran.py で指定した定数値（フォルダ名やファイル名）も一応インポートし，
# 本プログラムでも使えるようにしておく
# ここは基本的に変更しなくてOK
from train import MODEL_FILE


# テストデータセット（CSVファイル）のファイル名
# ここでは例として mobile_phone_test.csv を用い，
# 携帯端末のスペックから価格帯（0～3の4段階）を分類する分類器をテストする場合を例にとる
# mushroom_test.csv を用い，食べられるキノコか毒キノコかを分類する場合も試してみましょう．

actions = ["H", "D", "SR", "S"]
# ニューラルネットワーク net にデータ vec を入力して分類結果を得る関数


def predict(net, vec):

    net.eval()  # 必須．現状では「とりあえず記載しておく」という理解でOK

    x = torch.tensor([vec], device='cpu')  # 入力ベクトル vec をPyTorch用の型に変換
    z = F.softmax(net(x), dim=1)  # AIによる推定．Softmaxを適用して確率（っぽい値）を返すようにする
    z = z.to('cpu').detach().numpy().copy()[0]  # 結果をnumpy.ndarray 型に変換して返却
    z = np.argmax(z)
    return actions[z]


def output(data):
    # ニューラルネットワークの用意
    net = myMLP()  # 上で指定した名前（「myMLP」の部分）を指定する
    net.load_state_dict(torch.load(MODEL_FILE))  # 学習済みのパラメータをファイルからロードする

    # デバイスの指定とオプティマイザーの用意（基本このままでOK）
    device = 'cpu'
    net = net.to(device)

    # テストデータを入力して分類精度を評価
    net.eval()  # 必須．現状では「とりあえず記載しておく」という理解でOK
    return predict(net, data)
