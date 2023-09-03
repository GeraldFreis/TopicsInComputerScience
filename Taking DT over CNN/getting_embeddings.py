import pandas as pd

raw_data = pd.read_csv("IMDB_dataset_final.csv", low_memory=False)

# printing statements to check the variables
raw_x = raw_data.review; # print(raw_x)
raw_y = raw_data.sentiment; # print(raw_y)

number_of_words_in_dic = 37500; 

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=number_of_words_in_dic) # to tokenize the words for learning
tokenizer.fit_on_texts(raw_x)
tokenized_sentiments = tokenizer.texts_to_sequences(raw_x) # converting the words to number arrays

vocab_size = len(tokenizer.word_index) + 1

padded_tokenized_sentiments= tensorflow.keras.utils.pad_sequences(tokenized_sentiments, padding="post", maxlen=1000)

import keras
from keras import Sequential # we will be using this for the CNN
from keras import layers

embedding_layer = layers.Embedding(1,50,input_length=1000)
print(embedding_layer(padded_tokenized_sentiments[0]))
