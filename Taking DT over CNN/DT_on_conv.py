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

print("Creating Embedding and Convolutional layer Model")
model = k.Sequential()
model.add(CNN.get_layer("embedding"))
model.add(CNN.get_layer("conv1d"))
outputs = model(padded_tokenized_sentiments[0:50000:1])

# flattening the outputs
print("Up to flattening")
flatten = layers.Flatten()
outputs = flatten(outputs)

print("Up to np arraying")
outputs = np.array(outputs)

print("Up to splitting")
train_x, test_x, train_y, test_y = train_test_split(outputs, raw_y, random_state=1000, shuffle=True, test_size=0.3)

print("Fitting tree")
tree = DecisionTreeClassifier(max_depth=5).fit(train_x, train_y)

print("Predictions")
training_prediction = tree.predict(train_x)
test_prediction = tree.predict(test_x)

training_prediction_accuracy = sk.metrics.accuracy_score(train_y, training_prediction, normalize=True)
test_prediction_accuracy = sk.metrics.accuracy_score(test_y, test_prediction, normalize=True)


print("Training accuracy: {} vs Testing Accuracy {}".format(training_prediction_accuracy, test_prediction_accuracy))