import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000

def make_data_ready(Xtrain):
    tokenizer = Tokenizer(num_words=vocab_size,oov_token='<OOV>')
    tokenizer.fit_on_texts(Xtrain)

    # word_index = tokenizer.word_index

    trainsequences = tokenizer.texts_to_sequences(Xtrain)
    trainpadded = pad_sequences(trainsequences, padding='post', maxlen=100, truncating='post')
    return trainpadded


X = [
    "I absolutely love this product, it works perfectly every time.",
    "The service was amazing, and the staff was super helpful!",
    "This app is fantastic; it makes my life so much easier.",
    "Highly recommend this restaurant, the food was delicious and fresh.",
    "Great experience! I will definitely be coming back again soon.",
    "Wonderful customer support, resolved my issue very quickly!",
    "Fantastic performance, exceeded my expectations in every possible way.",
    "This is the best purchase Ive made all year!",
    "Smooth process from start to finish, very satisfied with everything.",
    "Top quality product, absolutely worth the price I paid.",
    "I absolutely hate this product; it never works as expected.",
    "The service was terrible, and the staff were unhelpful.",
    "This app is awful; it makes my life more difficult.",
    "I do not recommend this restaurant; the food was bland.",
    "Terrible experience! I will never return to this place again.",
    "Customer support was unresponsive, leaving my issue unresolved.",
    "Poor performance, did not meet my expectations in any way.",
    "This was the worst purchase Ive made this year!",
    "The process was frustrating from start to finish, very disappointing.",
    "Low quality product, not worth the price I paid."
]


Xready = make_data_ready(X)

loaded_model = tf.keras.models.load_model('mykerasmodel.keras')

loaded_model.summary()

pred = loaded_model.predict(Xready)

for i in range(len(X)):
    print(X[i] , " --------------- ", pred[i])