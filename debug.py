import pandas as pd
import sys

file_name = 'log/ai_player_log.csv' if len(
    sys.argv) < 2 else 'log/ai_model_player_log.csv'

df = pd.read_csv(file_name, header=None)
df = df[df[3] != 'unsettled']
win_count = len(df[df[3] == 'win'])
lose_count = len(df[df[3] == 'bust']) + len(df[df[3] == 'lose'])
draw_count = len(df[df[3] == 'draw'])

print('勝率:', win_count/(win_count+lose_count+draw_count))
print('敗率:', lose_count/(win_count+lose_count+draw_count))
