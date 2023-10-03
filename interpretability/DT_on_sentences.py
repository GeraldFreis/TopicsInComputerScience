"""Taking a DT over the sentences database and seeing if it performs better on smaller datasets"""

import sklearn
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("IMDB_sentences.csv")
sentences = df.Sentences
prediction = df.Sentence_CNN_prediction.astype(float)
m = prediction < 0.5
prediction.where(m, 1, inplace=True)
prediction.where(-m, 0, inplace=True)
print(prediction)
tokenizer = Tokenizer(len(sentences))
tokenizer.fit_on_texts(sentences)
tokenized_sentences = tokenizer.texts_to_sequences(sentences)
padded_sentences = tf.keras.utils.pad_sequences(tokenized_sentences, padding="post", maxlen=1000)

train_x, test_x, train_y, test_y = train_test_split(padded_sentences, prediction, shuffle=True, random_state=1000, test_size=0.3)

tree = DecisionTreeClassifier(max_depth=1800000).fit(train_x, train_y)

train_prediction = tree.predict(train_x)
test_prediction = tree.predict(test_x)

train_accuracy = sklearn.metrics.accuracy_score(train_prediction, train_y, normalize=True)
test_accuracy = sklearn.metrics.accuracy_score(test_prediction, test_y, normalize=True)

print("DT across training: {}\t DT across testing: {}\n".format(train_accuracy, test_accuracy))