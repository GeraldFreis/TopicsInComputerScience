import pandas as pd
# we want to take the sentences A = [sentences_n, for n in [0, len(sentences)/2]]

max_depth = 4 # by default the max depth of sentences interpretability we want to reach is 4, which means 2^4 (32) sentence depths

raw_paragraphs = pd.read_csv("IMDB.csv")

print(raw_paragraphs.review[0]) # this will be our first sentence to deconstruct

# splitting the sentence
sentences = raw_paragraphs.review[0].split(".")
sentences = [sentence for sentence in sentences if len(sentence) >= 2] # getting rid of empty sentences (for example elipses)

# print(sentences)

# creating out A and B subsets
A = [sentences[i] for i in range(0, int(len(sentences)/2))]
B = [sentences[i] for i in range(int(len(sentences)/2), len(sentences))]

string_A = str()
string_B = str()

for sentence in A:
    string_A += sentence + "."
for sentence in B:
    string_B += sentence + "."


# tokenizing the strings
import tensorflow as tf
import keras 
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max(len(string_A), len(string_B)))
tokenizer.fit_on_texts(string_A)
tokenized_A = tokenizer.texts_to_sequences(string_A)
tokenized_B = tokenizer.texts_to_sequences(string_B)

# padding the texts
padded_A = tf.keras.utils.pad_sequences(tokenized_A, maxlen=1000, padding="post")
padded_B = tf.keras.utils.pad_sequences(tokenized_B, maxlen=1000, padding="post")

# loading the CNN
model = keras.models.load_model("../CNN_Non_Dense")

# running the strings through the CNN and getting the prediction
prediction_A = model.predict(padded_A, verbose=1)
prediction_B = model.predict(padded_B, verbose=1)

avg_A = float(sum(prediction_A)) / float(len(prediction_A))
avg_B = float(sum(prediction_B)) / float(len(prediction_B)) 

print("Prediction for A: {}\nPrediction for B: {}".format(avg_A, avg_B))

