import pandas as pd
import tensorflow as tf
import keras as k
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras import layers
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

tf.config.set_visible_devices([], 'GPU') # turning GPU use off as tensors exceed 10000*1000*50

# importing the raw data
raw_data = pd.read_csv("IMDB_with_predictions.csv")
raw_x = raw_data.review
raw_y = raw_data.CNN_Predictions

# tokenizing the inputs
tokenizer = Tokenizer(num_words=37500) # to tokenize the words for learning
tokenizer.fit_on_texts(raw_x)
tokenized_sentiments = tokenizer.texts_to_sequences(raw_x) # converting the words to number arrays

vocab_size = len(tokenizer.word_index) + 1

padded_tokenized_sentiments= tf.keras.utils.pad_sequences(tokenized_sentiments, padding="post", maxlen=1000)

print("Loading CNN")
CNN = tf.keras.models.load_model("../CNN_Non_Dense")

print("Creating Embedding layer Model")
embeddings = k.Model(inputs=CNN.inputs, outputs=CNN.get_layer(name="embedding").output)
embedding_raw_outputs = embeddings(padded_tokenized_sentiments[0:50000:1])

print("Up to flattening")
flatten_layer = k.layers.Flatten()
embedding_outputs = flatten_layer(embedding_raw_outputs)

print("Up to converting to NP arrays")
embedding_outputs = np.array(embedding_outputs)

print("Up to splitting")
train_embedding_x, test_embedding_x, train_embedding_y, test_embedding_y = train_test_split(embedding_outputs, raw_y, random_state=1000, shuffle=True, test_size=0.3)

print("Up to fitting tree")
embedding_tree = DecisionTreeClassifier(max_depth=5).fit(train_embedding_x, train_embedding_y)

print("Up to getting training and test predictions")
training_embedding_prediction = embedding_tree.predict(train_embedding_x)
test_embedding_prediction = embedding_tree.predict(test_embedding_x)

print("Up to getting accuracy")
# getting accuracies
training_embedding_prediction_accuracy = sk.metrics.accuracy_score(train_embedding_y, training_embedding_prediction, normalize=True)
test_embedding_prediction_accuracy = sk.metrics.accuracy_score(test_embedding_y, test_embedding_prediction, normalize=True)

print("Embedding layer\nTraining accuracy: {} vs Testing Accuracy {}".format(training_embedding_prediction_accuracy, test_embedding_prediction_accuracy))