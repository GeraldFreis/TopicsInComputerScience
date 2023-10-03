import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import keras
from keras import Sequential
from keras import layers
import os
import sklearn as sk
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier

raw_data = pd.read_csv("IMDB_with_predictions.csv", low_memory=False)

# printing statements to check the variables
raw_x = raw_data.review; # print(raw_x)
raw_y = raw_data.CNN_Predictions; # print(raw_y)

number_of_words_in_dic = 37500; 
print("Up to tokenizing")
# getting the words to be workable
tokenizer = Tokenizer(num_words=number_of_words_in_dic) # to tokenize the words for learning
tokenizer.fit_on_texts(raw_x)
tokenized_sentiments = tokenizer.texts_to_sequences(raw_x) # converting the words to number arrays

vocab_size = len(tokenizer.word_index) + 1

padded_tokenized_sentiments= tf.keras.utils.pad_sequences(tokenized_sentiments, padding="post", maxlen=1000)
tf.config.set_visible_devices([], 'GPU') # turning GPU use off as tensors exceed 10000*1000*50


# loading the keras model
CNN = tf.keras.models.load_model("CNN_Non_Dense")
print("Up to setting up output models {to extract output}")
# getting layer outputs
embeddings = keras.Model(inputs=CNN.inputs, outputs=CNN.get_layer(name="embedding").output) 
conv = keras.Model(inputs=CNN.inputs, outputs=CNN.get_layer(name="conv1d").output) 
pooling = keras.Model(inputs=CNN.inputs, outputs=CNN.get_layer(name="global_max_pooling1d").output) 
dense = keras.Model(inputs=CNN.inputs, outputs=CNN.get_layer(name="dense").output) 
dense_1 = keras.Model(inputs=CNN.inputs, outputs=CNN.get_layer(name="dense_1").output) 
print("Up to getting the outputs")
# getting the raw outputs when passed the tokenized sentiments
embedding_raw_outputs = embeddings(padded_tokenized_sentiments[0:50000:1])
conv_raw_out = conv(padded_tokenized_sentiments[0:50000:1])
pooling_raw_out = pooling(padded_tokenized_sentiments[0:50000:1])
dense_raw_out = dense(padded_tokenized_sentiments[0:50000:1])
dense_1_raw_out = dense_1(padded_tokenized_sentiments[0:50000:1])

flatten_layer = keras.layers.Flatten()
print("Up to flattening")
# flattening the outputs of each layer
embedding_outputs = flatten_layer(embedding_raw_outputs)
conv_outputs = flatten_layer(conv_raw_out)
pooling_outputs = flatten_layer(pooling_raw_out)
dense_outputs = flatten_layer(dense_raw_out)
dense_1_outputs = flatten_layer(dense_1_raw_out)
print("Up to converting to NP arrays")
# converting each output to a numpy array
embedding_outputs = np.array(embedding_outputs)
conv_outputs = np.array(conv_outputs)
pooling_outputs = np.array(pooling_outputs)
dense_outputs = np.array(dense_outputs)
dense_1_outputs = np.array(dense_1_outputs)
print("Up to splitting")
# splitting the outputs of each layer
train_embedding_x, test_embedding_x, train_embedding_y, test_embedding_y = train_test_split(embedding_outputs, raw_y, random_state=1000, shuffle=True, test_size=0.3)
train_conv_x, test_conv_x, train_conv_y, test_conv_y = train_test_split(conv_outputs, raw_y, random_state=1000, shuffle=True, test_size=0.3)
train_pooling_x, test_pooling_x, train_pooling_y, test_pooling_y = train_test_split(pooling_outputs, raw_y, random_state=1000, shuffle=True, test_size=0.3)
train_dense_x, test_dense_x, train_dense_y, test_dense_y = train_test_split(dense_outputs, raw_y, random_state=1000, shuffle=True, test_size=0.3)
train_dense_1_x, test_dense_1_x, train_dense_1_y, test_dense_1_y = train_test_split(dense_1_outputs, raw_y, random_state=1000, shuffle=True, test_size=0.3)
print("Up to fitting tree")
# predicting each of the layers to the CNN's predictions using DT
embedding_tree = DecisionTreeClassifier(max_depth=5).fit(train_embedding_x, train_embedding_y)
conv_tree = DecisionTreeClassifier(max_depth=5).fit(train_conv_x, train_conv_y)
pooling_tree = DecisionTreeClassifier(max_depth=5).fit(train_pooling_x, train_pooling_y)
dense_tree = DecisionTreeClassifier(max_depth=5).fit(train_dense_x, train_dense_y)
dense_1_tree = DecisionTreeClassifier(max_depth=5).fit(train_dense_1_x, train_dense_1_y)

print("Up to getting training and test predictions")
# getting the training and test prediction for each layer
training_embedding_prediction = embedding_tree.predict(train_embedding_x)
test_embedding_prediction = embedding_tree.predict(test_embedding_x)

training_conv_prediction = conv_tree.predict(train_conv_x)
test_conv_prediction = conv_tree.predict(test_conv_x)

training_pooling_prediction = pooling_tree.predict(train_pooling_x)
test_pooling_prediction = pooling_tree.predict(test_pooling_x)

training_dense_prediction = dense_tree.predict(train_dense_x)
test_dense_prediction = dense_tree.predict(test_dense_x)

training_dense_1_prediction = dense_1_tree.predict(train_dense_1_x)
test_dense_1_prediction = dense_1_tree.predict(test_dense_1_x)
print("Up to getting accuracies")
# getting accuracies
training_embedding_prediction_accuracy = sk.metrics.accuracy_score(train_embedding_y, training_embedding_prediction, normalize=True)
test_embedding_prediction_accuracy = sk.metrics.accuracy_score(test_embedding_y, test_embedding_prediction, normalize=True)

training_conv_prediction_accuracy = sk.metrics.accuracy_score(train_conv_y, training_conv_prediction, normalize=True)
test_conv_prediction_accuracy = sk.metrics.accuracy_score(test_conv_y, test_conv_prediction, normalize=True)

training_pooling_prediction_accuracy = sk.metrics.accuracy_score(train_pooling_y, training_pooling_prediction, normalize=True)
test_pooling_prediction_accuracy = sk.metrics.accuracy_score(test_pooling_y, test_pooling_prediction, normalize=True)

training_dense_prediction_accuracy = sk.metrics.accuracy_score(train_dense_y, training_dense_prediction, normalize=True)
test_dense_prediction_accuracy = sk.metrics.accuracy_score(test_dense_y, test_dense_prediction, normalize=True)

training_dense_1_prediction_accuracy = sk.metrics.accuracy_score(train_dense_1_y, training_dense_1_prediction, normalize=True)
test_dense_1_prediction_accuracy = sk.metrics.accuracy_score(test_dense_1_y, test_dense_1_prediction, normalize=True)

# printing accuracies
print("Embedding layer\nTraining accuracy: {} vs Testing Accuracy {}".format(training_embedding_prediction_accuracy, test_embedding_prediction_accuracy))
print("Convolution layer\nTraining accuracy: {} vs Testing Accuracy {}".format(training_conv_prediction_accuracy, test_conv_prediction_accuracy))
print("Pooling layer\nTraining accuracy: {} vs Testing Accuracy {}".format(training_pooling_prediction_accuracy, test_pooling_prediction_accuracy))
print("Dense layer\nTraining accuracy: {} vs Testing Accuracy {}".format(training_dense_prediction_accuracy, test_dense_prediction_accuracy))
print("Dense 2nd layer\nTraining accuracy: {} vs Testing Accuracy {}".format(training_dense_1_prediction_accuracy, test_dense_1_prediction_accuracy))