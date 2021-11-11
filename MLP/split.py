import sys
import csv
import numpy as np


# CSVファイルのエンコード形式
# デフォルトでは UTF-8 を想定する
# ※ 日本語を含まないCSVファイルを扱うときは FILE_ENCODING = None とするのが良いです．
# ※ Windows で日本語を含むCSVファイルを扱うときは FILE_ENCODING = 'shift_jis' とすると正常に動作するかもしれません．
FILE_ENCODING = 'utf_8'


# CSVファイルを読み込む関数
def read_csv(filename, encoding):
    f = open(filename, 'r', encoding=encoding)
    reader = csv.reader(f)
    header = next(reader)
    data = []
    for row in reader:
        data.append(row)
    f.close()
    return header, data


# 一行分のデータをファイルに出力する関数
def print_row(row, of):
    k = len(row)
    print(row[0], file=of, end='')
    for i in range(1, k):
        print(',{0}'.format(row[i]), file=of, end='')
    print(file=of)


if len(sys.argv) < 3:
    print('usage: python split.py xxxx.csv n')
    print('\txxxx.csv: データセット（正解付き事例の全集合）を記載したCSVファイルのファイル名')
    print('\tn: テストデータとして使用するデータの数')
    print('')
    print('例えば hoge.csv から 1000 個を取り出してテストデータセットとして，')
    print('残りを学習データセットとして使用したい場合は，次のように実行して下さい．')
    print('python split.py hoge.csv 1000')
    print('')
    print('なお，CSVファイルは，先頭行が項目名を記載する行となるように作成して下さい．')
    print('')
    exit()

filename = sys.argv[1] # CSVファイル名
n_test_samples = int(sys.argv[2]) # テストデータの数

header, data = read_csv(filename, encoding=FILE_ENCODING) # CSVファイル読込

n_total = len(data) # 全データ数

# 出力用のファイルを作成
p = filename.rfind('.')
out_file_test = filename[:p] + '_test.csv'
out_file_train = filename[:p] + '_train.csv'
perm = np.random.permutation(n_total)
of = open(out_file_test, 'w', encoding=FILE_ENCODING)
print_row(header, of)
for i in range(n_test_samples):
    print_row(data[perm[i]], of)
of.close()
of = open(out_file_train, 'w', encoding=FILE_ENCODING)
print_row(header, of)
for i in range(n_test_samples, n_total):
    print_row(data[perm[i]], of)
of.close()
