import pandas as pd
import sklearn as sk

data_raw = pd.read_csv("IMDB_with_predictions.csv", low_memory=False)
raw_x = data_raw.review;
raw_y = data_raw.CNN_Predictions;

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(raw_x, raw_y, random_state=1000, shuffle=True, test_size=0.25)


# either positive or negative sentiment
import tensorflow

from tensorflow.keras.preprocessing.text import Tokenizer
# tokenizing the input
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_x)
tokenized_raw_text = tokenizer.texts_to_sequences(raw_x)

padded_raw_text = tensorflow.keras.utils.pad_sequences(tokenized_raw_text, padding="post", maxlen=200)

# new tokenizer for the other training splits
# creating a validation sample from the training dataset
train_x, validation_x, train_y, validation_y = sk.model_selection.train_test_split(train_x, train_y, random_state=1000, shuffle=True, test_size=0.2)

proper_tokenizer = Tokenizer()
proper_tokenizer.fit_on_texts(train_x)
proper_tokenizer.fit_on_texts(test_x)
proper_tokenizer.fit_on_texts(validation_x)

tokenized_train_x = proper_tokenizer.texts_to_sequences(train_x);
tokenized_test_x = proper_tokenizer.texts_to_sequences(test_x)
tokenized_validation_x = proper_tokenizer.texts_to_sequences(validation_x)
# ensuring equal length to all sequences
padded_train_x = tensorflow.keras.utils.pad_sequences(tokenized_train_x, padding="post", maxlen=200)
padded_test_x = tensorflow.keras.utils.pad_sequences(tokenized_test_x, padding="post", maxlen=200)
padded_validation_x = tensorflow.keras.utils.pad_sequences(tokenized_validation_x, padding="post", maxlen=200)

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
# plt.rcParams.update({'font.size': 32})

plt.rcParams['figure.dpi']=200
raw_tree = DecisionTreeClassifier(max_depth=3).fit(padded_raw_text, raw_y)

# plot_tree(raw_tree, filled=True)
# plt.title("Decision tree trained on all the treeeee features")
# plt.show()
accuracy_list = list()
training_accuracy_list = list()
best_test_accuracy = 0
best_i = 0
# evaluating the models
for i in range(1, 30):
    tree = DecisionTreeClassifier(max_depth=i).fit(padded_train_x, train_y)
    training_prediction = tree.predict(padded_train_x)
    test_prediction = tree.predict(padded_test_x)

    training_prediction_accuracy = sk.metrics.accuracy_score(train_y, training_prediction, normalize=True)
    test_prediction_accuracy = sk.metrics.accuracy_score(test_y, test_prediction, normalize=True)
    accuracy_list.append(test_prediction_accuracy)
    training_accuracy_list.append(training_prediction_accuracy)

print("Training accuracy: {} vs Testing Accuracy {}".format(training_prediction_accuracy, test_prediction_accuracy))
plt.plot(accuracy_list, label="Testing")
plt.plot(training_accuracy_list, label="Training")
plt.legend()

plt.title("Accuracy vs Depth")
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.show()