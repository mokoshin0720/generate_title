from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess as preprocess
import data_gen
import preprocessing

with open("document_list.txt", "w") as f:
    for i in range(len(data_gen.data)):
        tmp = preprocessing.make_wakati(data_gen.data["title"][i])
        tmp = " ".join(tmp)
        f.write(tmp + "\n")

f = open('document_list.txt','r')

trainings = [TaggedDocument(words = data.split(),tags = [i]) for i,data in enumerate(f)]
doc2vec_model = Doc2Vec(documents= trainings, vector_size=400, windows=5, min_count=5, epochs=100)