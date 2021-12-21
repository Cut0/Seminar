import numpy as np
import torch
import torch.nn.functional as F
from train import myMLP
from train import MODEL_FILE

actions = ["H", "D", "SR", "S"]


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
