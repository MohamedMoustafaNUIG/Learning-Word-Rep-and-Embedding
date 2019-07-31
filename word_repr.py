import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras import backend as K
import warnings
warnings.simplefilter('ignore')

def generate_numerical_repr(text, vocabulary_dict, pad_size=None):
    size_of_vocab = len(vocabulary_dict.keys())
    tokens_list = text.split(" ")
    if pad_size:
        n_of_tokens = pad_size
        n_of_columns = size_of_vocab + 2
    else:
        n_of_tokens = len(tokens_list)
        n_of_columns = size_of_vocab + 2

    text_num_repr = np.zeros([n_of_tokens, n_of_columns])
    tokensIdx_to_considerate = min(len(tokens_list), pad_size) \
        if pad_size else len(tokens_list)
    for token_pos in range(0, tokensIdx_to_considerate):
        token = tokens_list[token_pos]
        column_pos = vocabulary_dict.get(token, size_of_vocab)
        text_num_repr[token_pos][column_pos] = 1
    if pad_size:
        for endSentence_pos in range(token_pos+1, pad_size):
            column_pos = size_of_vocab + 1
            text_num_repr[endSentence_pos][column_pos] = 1
    return text_num_repr

sample_text_1 = "bitty bought a bit of butter"
sample_text_2 = "but the bit of butter was a bit bitter"
sample_text_3 = \
        "so she bought some better butter to make the bitter butter better"
corpus = [sample_text_1, sample_text_2, sample_text_3]
no_docs = len(corpus)

vocabulary = {}
for document in corpus:
    split_document = document.split(" ")
    for token in split_document:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
print(vocabulary)
size_of_vocabulary = len(vocabulary.keys())

int_repr_corpus =[]
for sentence in corpus:
	sen_list = []
	for token in sentence.split(" "):
		sen_list.append(vocabulary[token])
	int_repr_corpus.append(sen_list)
print(int_repr_corpus)

one_hot_encoding_corpus = np.zeros([no_docs, size_of_vocabulary])
for document_idx, document in enumerate(corpus):
    for token_idx, token in enumerate(document.split(" ")):
        one_hot_encoding_corpus[document_idx, token_idx] = 1
print(one_hot_encoding_corpus)

for document in corpus:
    n_repr = generate_numerical_repr(document, vocabulary, pad_size=None)
    print(document)
    print(n_repr)
    print(n_repr.shape)
    input()

for document in corpus:
    n_repr = generate_numerical_repr(document, vocabulary, pad_size=5)
    print(document)
    print(n_repr)
    print(n_repr.shape)
    input()

docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!', 'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']

# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])
docs = [d.lower().replace("!", "").replace(".", "") for d in docs]
vocab = CountVectorizer()
vocab.fit(docs)

encoded_docs = [[vocab.vocabulary_.get(w, 0) for w in doc.split(" ")] for doc in docs]

print(encoded_docs)
print("\n")

# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

model = Sequential()
model.add(Embedding(50, 8, input_length=4))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=['acc'])

model.fit(padded_docs, labels, epochs=50, verbose=0)
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy :', accuracy*100)