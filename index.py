# Model Trainer

#importing modules
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re

# function for cleaning text with unecessary characters
def clean_text(text):
    try:
        text = re.sub(r'Ã[\x80-\xBF]+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text.lower()
    except:
        print(text)

testtrasinsplit = 35000
vocab_size = 10000

# reading data from a CSV file
csvdataset = pd.read_csv("./DATASET.csv")
csvdataset = csvdataset.dropna()
# csvdataset['Review'] = csvdataset['Review'].apply(clean_text)
X = csvdataset.iloc[:, 0]
Y = csvdataset.iloc[:, 1]
Y = list(map(lambda x: 1 if x == "POSITIVE" else 0, Y))

# Splitting the data into training and testing sets
Xtrain = X[:testtrasinsplit]
Ytrain = Y[:testtrasinsplit]
Xtest = X[testtrasinsplit:]
Ytest = Y[testtrasinsplit:]

# Initializing the tokenizer
tokenizer = Tokenizer(num_words=vocab_size,oov_token='<OOV>')
tokenizer.fit_on_texts(Xtrain)

# word_index = tokenizer.word_index   # to show word indexes (optional)

trainsequences = tokenizer.texts_to_sequences(Xtrain)
trainpadded = pad_sequences(trainsequences, padding='post', maxlen=100, truncating='post')
testsequences = tokenizer.texts_to_sequences(Xtest)
testpadded = pad_sequences(testsequences, padding='post', maxlen=100, truncating='post')

trainpadded = np.array(trainpadded)
trainlabels = np.array(Ytrain)
testpadded = np.array(testpadded)
testlabels = np.array(Ytest)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

num_epochs = 15
model.fit(trainpadded, trainlabels, epochs=num_epochs, validation_data=(testpadded, testlabels), verbose=2)

# model can be saved either as .h5 or .keras
# model.save('mymodel.h5')
model.save('mykerasmodel.keras')