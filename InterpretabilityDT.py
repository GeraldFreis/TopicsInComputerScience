# this model will just try to predict the output of the CNN based on the paragraph and associated sentiment
import pandas as pd
import keras # 2.11.0
import tensorflow # 2.11.0
import sklearn as sk

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


raw_data = pd.read_csv("IMDB_with_predictions.csv")
# raw_data.drop(columns=["Unnamed: 0"], inplace=True)  # this is only required if the dataset messes up
x = raw_data.review
# print(len(x))
y = raw_data.CNN_Predictions
prediction_train = raw_data.sentiment[0:40000:1]
prediction_test = raw_data.sentiment[40000:len(raw_data.sentiment):1]

# print(len(prediction_train))
# print(len(prediction_test))

train_x, test_x, train_y, test_y = train_test_split(x,y,random_state=1000, test_size=0.2, shuffle=True)

# tokenizing the training and testing subsets because we need the data in a processable form for the ML Model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
tokenized_train_x = tokenizer.texts_to_sequences(train_x)
tokenized_test_x = tokenizer.texts_to_sequences(test_x)

# padding the sequences now to ensure equal sizing
padded_tokenized_train_x = tensorflow.keras.utils.pad_sequences(tokenized_train_x, padding="post", maxlen=1000)
padded_tokenized_test_x = tensorflow.keras.utils.pad_sequences(tokenized_test_x, padding="post", maxlen=1000)
# padded_tokenized_train_x[0][999] = prediction_train[0]
# print(padded_tokenized_train_x[0])
# maxl = len(padded_tokenized_train_x[0])

# problem: cannot nicely train two columns because we need one to be tokenized. To solve we place the associated prediction
# -> at the n-1th value of the tokenized text; hence simulating that it is another column
for i in range(0, len(padded_tokenized_train_x)):
    padded_tokenized_train_x[i][999] = prediction_train[i]
for i in range(0, len(padded_tokenized_test_x)):
    padded_tokenized_test_x[i][999] = prediction_test[40000+i] # 40000 + i because prediction_test retained its original indexs

# print(padded_tokenized_train_x)
# # training the DT to predict what the CNN does
from sklearn.tree import DecisionTreeClassifier

best_test_accuracy = 0
best_i = 0
# evaluating the models
for i in range(1, 20):
    tree = DecisionTreeClassifier(max_depth=i).fit(padded_tokenized_train_x, train_y)
    training_prediction = tree.predict(padded_tokenized_train_x)
    test_prediction = tree.predict(padded_tokenized_test_x)

    training_prediction_accuracy = sk.metrics.accuracy_score(train_y, training_prediction, normalize=True)
    test_prediction_accuracy = sk.metrics.accuracy_score(test_y, test_prediction, normalize=True)


    print("{}: Training accuracy: {} vs Testing Accuracy {}".format(i, training_prediction_accuracy, test_prediction_accuracy))
    if(test_prediction_accuracy > best_test_accuracy): 
        best_test_accuracy = test_prediction_accuracy
        best_i=i

print("Best depth was: {} with {}".format(best_i, best_test_accuracy))