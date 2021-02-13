import numpy as np
import nltk
import torch
import model

encoder = Encoder(vocab_size, embedding_dim, hidden_dim, batch_size=1).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim, batch_size=1).to(device)

text = title_train # title_train / title_test

for epoch in range(35, 60, 10):
    model_name = "title{}.pt".format(epoch)
    checkpoint = torch.load(model_name)
    encoder.load_state_dict(checkpoint["encoder_model"])
    decoder.load_state_dict(checkpoint["decoder_model"])

    print("Checkpoint {:>3d}".format(epoch))
    print("-"*30)
    accuracy = 0
    BLEU_score = 0
    sim_score = 0

    with torch.no_grad():
        for i in range(len(text)):
            for cnt in range(5):
                x = text[i]
                input_tensor = torch.tensor([x], device=device)
                mu, logvar, z, encoder_hidden = encoder(input_tensor)
                decoder_hidden = encoder_hidden
                token = "<eos>"

                title = []
                for _ in range(title_size):
                    index = word2id[token]
                    input_tensor = torch.tensor([index], device=device)
                    output, decoder_hidden = decoder(input_tensor, decoder_hidden)
                    prob = F.softmax(torch.squeeze(output))
                    index = torch.argmax(prob.cpu().detach()).item()
                    token = id2word[index]
                    if token == "<eos>":
                        break
                    title.append(token)

                t = text[i]
                t = [s for s in t if s!= 1 and s != 0]
                reference = []
                for w in t:
                    reference.append(id2word[w])

                # BLEU Score
                BLEU_tmp = nltk.translate.bleu_score.sentence_bleu([reference], title)
                BLEU_score += BLEU_tmp

                # Doc2Vec Score
                sim_tmp = doc2vec_model.docvecs.similarity_unseen_docs(doc2vec_model, reference, title, alpha=1, min_alpha=0.0001, steps=100)
                sim_score += sim_tmp

                # if reference != title:
                #     if cnt < 30:
                #         print("出力", "".join(title))
                #         # print(sim_tmp)
                #     if cnt == 29:
                #         print("入力", "".join(reference))
                #         print("="*50)

    output_num = len(text) * 5
    print("BLEU Score: {:.5f}".format(BLEU_score / output_num))
    print("cos Score: {:.5f}".format(sim_score / output_num))
    print("-"*30)