import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend as K
import warnings
warnings.simplefilter('ignore')

np.random.seed(7)
#want only most frequent 5000 words
top_words = 5000
#save current load configs
np_load_old = np.load
#change load configs
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
#load data using new configs
(X_train , y_train), (X_test , y_test) = imdb.load_data(num_words = top_words)
#reset configs for future usage
np.load = np_load_old
#add padding or remove words to make each review 500 tokens long
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_length)

#create model architecture
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
print(model.summary())

#train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64)

#evaluate model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy : ", scores[1]*100)
