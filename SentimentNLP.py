import pandas as pd
# i like to do running imports because it is fun fr

raw_data = pd.read_csv("IMDB_dataset_final.csv", low_memory=False)

# printing statements to check the variables
raw_x = raw_data.review; # print(raw_x)
raw_y = raw_data.sentiment; # print(raw_y)

from sklearn.model_selection import train_test_split

training_x, testing_x, training_y, testing_y = train_test_split(raw_x, raw_y, shuffle=True, random_state=1000, test_size=0.25)

# printing statements to check variables
# print(training_x); print(len(training_x))
# print(testing_x); print(len(testing_x))

number_of_words_in_dic = 37500; # 90000 semi common words in the english language

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=number_of_words_in_dic) # to tokenize the words for learning
tokenizer.fit_on_texts(training_x)
tokenized_training_x = tokenizer.texts_to_sequences(training_x)
tokenized_testing_x = tokenizer.texts_to_sequences(testing_x)
print(tokenized_training_x[0])
vocab_size = len(tokenizer.word_index) + 1

padded_tokenized_training_x = tensorflow.keras.utils.pad_sequences(tokenized_training_x, padding="post", maxlen=1000)
padded_tokenized_testing_x = tensorflow.keras.utils.pad_sequences(tokenized_testing_x, padding="post", maxlen=1000)

import keras
from keras import Sequential # we will be using this for the CNN
from keras import layers

CNN = Sequential()
CNN.add(layers.Embedding(37500, 50, input_length=1000)) # given by max word len, embedding dim and max len
CNN.add(layers.Conv1D(32, 3, activation="relu")) # convolutional layer
CNN.add(layers.GlobalMaxPooling1D(trainable=True, dynamic=False))
CNN.add(layers.Dense(10, activation="relu"))
CNN.add(layers.Dense(1, activation="sigmoid")) # final layer of the CNN

# compiling CNN
CNN.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

training = CNN.fit(padded_tokenized_training_x, training_y, verbose=True, validation_data=(padded_tokenized_testing_x, testing_y), batch_size=10, epochs=4)

# testing and training
loss_training, accuracy_training = CNN.evaluate(padded_tokenized_training_x, training_y, verbose=True)
loss_testing, accuracy_testing = CNN.evaluate(padded_tokenized_testing_x, testing_y, verbose=True)

print("Training vs Testing accuracy: {} vs {}".format(accuracy_training, accuracy_testing))

CNN.save("Trained_CNN_for_sentiment_prediction")
# prediction_training
