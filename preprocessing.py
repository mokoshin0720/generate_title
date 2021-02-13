import MeCab
import re
import torch
import data_gen

# 分かち書きの関数
tagger = MeCab.Tagger("-Owakati")

def make_wakati(sentence):
    sentence = tagger.parse(sentence)
    sentence = re.sub('[0-9０-９]+', "0", sentence)
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./→←○\n\u3000]+', "", sentence)
    wakati = sentence.split(" ")
    wakati = list(filter("".__ne__, wakati))
    return wakati

# 辞書の作成
word2id = {"<pad>": 0, "<eos>": 1}

for title in data_gen.data["title"]:
    wakati = make_wakati(title)
    for word in wakati:
        if word in word2id: continue
        word2id[word] = len(word2id)

id2word = {v: k for k, v in word2id.items()}

print("vocab size: ", len(word2id))
print(word2id)
print("vocab size: ", len(id2word))
print(id2word)

# 文章からIDを出力する関数
def sentence2id(sentence):
    wakati = make_wakati(sentence)
    wakati.insert(0, "<eos>")
    wakati.insert(len(wakati), "<eos>")
    return [word2id[w] for w in wakati]

print(data_gen.data["title"][1])
print(sentence2id(data_gen.data["title"][1]))

# 最大単語数の取得
title_size = 0
for i in range(len(data_gen.data)):
    if title_size < len(sentence2id(data_gen.data["title"][i])):
        title_size = len(sentence2id(data_gen.data["title"][i]))
print(title_size)

# Paddingの挿入
def add_padding(sentence, size):
    while len(sentence) < size:
        sentence.append(0)
    return sentence

title_list = []

def make_data():
    if title_list == []:
        for i in range(len(data_gen.data)):
            title_list.append(add_padding(sentence2id(data_gen.data["title"][i]), title_size))
        return title_list
    else:
        return title_list

make_data()
print(title_list)

# 訓練データ、テストデータに分割
from sklearn.model_selection import train_test_split

make_data()
title_train, title_test = train_test_split(title_list, test_size=0.2)

print(title_train[0])
print(title_test[0])

# データをミニバッチに分ける
from sklearn.utils import shuffle

batch_size = 100

def train2batch(data, batch_size):
    title_batch = []
    data = shuffle(data) # dataをランダムに並び替え

    for i in range(0, len(data), batch_size):
        input_tmp = []
        output_tmp = []
        # print(i)
        if i > 7167:
            break

        for j in range(i, i+batch_size):
            input_tmp.append(data[j])

        title_batch.append(input_tmp)

    return title_batch

title_batch = train2batch(title_train, batch_size)
print(title_batch)
print(len(title_batch))

# Word-drop-out
import random
import copy

def rand_ints_nodup(a, b, k):
  ns = []
  while len(ns) < k:
    n = random.randint(a, b)
    if not n in ns:
      ns.append(n)
  return ns

def word_drop_out(batch):
    for title in batch:
        rand = rand_ints_nodup(1, len(title)-1, int(len(title)*0.4))
        for index in rand:
            if title[index] != word2id["<eos>"]:
                title[index] = word2id["<pad>"]
    return batch