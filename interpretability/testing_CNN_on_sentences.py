import pandas as pd
import keras as k
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# importing the dataset and the CNN 
raw_data = pd.read_csv("IMDB_sentences_broken.csv")
# raw_data.drop(columns=["Unnamed"])
print(raw_data)
sentences = raw_data.Sentences

CNN = tf.keras.models.load_model("../CNN_Non_Dense")

# tokenizing and padding all of the sentences
tokenizer = Tokenizer(num_words=len(sentences)) # to tokenize the words for learning
tokenizer.fit_on_texts(sentences)
tokenized_sentences = tokenizer.texts_to_sequences(sentences)
padded_sentences = tf.keras.utils.pad_sequences(tokenized_sentences, padding="post", maxlen=1000)
print(len(padded_sentences[0]))
print(len(padded_sentences[1]))
print(len(padded_sentences))
# feeding the sentences into the CNN and then getting the predictions
predictions = CNN.predict(padded_sentences, verbose=1, workers=4)
print(predictions[2])
raw_data["Sentence_CNN_prediction"] = predictions
# raw_data.to_csv("IMDB_sentences_broken_with_predictions.csv", index=False)