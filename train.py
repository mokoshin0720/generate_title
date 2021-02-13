from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import preprocessing
import model

# Encoder, Decoderの初期化
encoder = model.Encoder(vocab_size, embedding_dim, hidden_dim, batch_size).to(device)
decoder = model.Decoder(vocab_size, embedding_dim, hidden_dim, batch_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word2id["<pad>"], size_average=False)

# 最適化関数の定義
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("学習開始")
n_epoch = 60
sig_cnt = 0
all_losses = []

for epoch in range(1, n_epoch+1):
    title_batch = preprocessing.train2batch(title_train, batch_size)
    
    if epoch > 25:
        sig_cnt += 1

    for i in range(len(title_batch)):
        # 勾配を0で埋める → 勾配の初期化的な
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # データをテンソルに変換する
        title = torch.tensor(title_batch[i], device=device)
        # word_drop_out
        tmp_drop = copy.deepcopy(title_batch[i])
        drop_title = preprocessing.word_drop_out(tmp_drop)
        drop_torch = torch.tensor(drop_title, device=device)

        # Encoderにデータを渡す -> mu, logvar, z, encoder_hiddenの獲得
        mu, logvar, _, encoder_hidden = encoder(title)
        decoder_hidden = encoder_hidden

        source = drop_torch[:, :-1] # Deoderへの入力データ
        target = title[:, 1:] # Decoderの出力データ

        loss = 0
        for j in range(source.size(1)):
            decoder_output, decoder_hidden = decoder(source[:, j], decoder_hidden)
            decoder_output = torch.squeeze(decoder_output)
            reconst_loss = criterion(decoder_output, target[:, j])
            loss += reconst_loss

        sigmoid = lambda x : 1 / (1 + np.exp(-x))
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if epoch > 24:
            sig = sigmoid(-4 + sig_cnt*0.26)
            loss += kld * sig
            print("sigmoid: ", sig)
        else:
            all_losses.append(loss)

        print(kld)
        print("="*100)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    print(get_current_time(), "Epoch %d: loss(%.2f)" % (epoch, loss.item()))

    if epoch % 5 == 0:
        model_name = "title{}.pt".format(epoch)
        torch.save({
            'encoder_model': encoder.state_dict(),
            'decoder_model': decoder.state_dict(),
        }, model_name)
        print("Saving the checkpoint...\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


# 損失関数の可視化
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(all_losses)