import pandas as pd

daigo = pd.read_csv("/content/daigo.csv")
syacho = pd.read_csv("/content/はじめしゃちょー.csv")
hikaru = pd.read_csv("/content/ヒカル.csv")
bondo = pd.read_csv("/content/水溜りボンド.csv")
maataso = pd.read_csv("/content/まあたそ.csv")
kaji = pd.read_csv("/content/カジサック.csv")
ferumi = pd.read_csv("/content/フェルミ研究所.csv")
tokai = pd.read_csv("/content/東海オンエア.csv")
hikakin = pd.read_csv("/content/ヒカキン.csv")
data = pd.concat([daigo, syacho, maataso, kaji, ferumi, tokai, hikaru, bondo, hikakin], ignore_index=True)[:14000]
data