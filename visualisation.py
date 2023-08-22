import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# testing the different DT's on ground vs false truth
ground_df = pd.read_csv("IMDB_dataset_final.csv", low_memory=False)
false_df = pd.read_csv("IMDB_with_predictions.csv", low_memory=False)

ground_train_x, ground_test_x, ground_train_y, ground_test_y = train_test_split(ground_df.review, ground_df.sentiment, random_state=1000, shuffle=True, test_size=0.25)
false_train_x, false_test_x, false_train_y, false_test_y = train_test_split(false_df.review, false_df.CNN_Predictions, random_state=1000, shuffle=True, test_size=0.25)

# Tokenizing each text because the DT can only work with numeric values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(ground_df.review) # fitting on review as it is consistent across both
tokenized_ground_train = tokenizer.texts_to_sequences(ground_train_x)
tokenized_ground_test = tokenizer.texts_to_sequences(ground_test_x)

tokenized_false_train = tokenizer.texts_to_sequences(false_train_x)
tokenized_false_test = tokenizer.texts_to_sequences(false_test_x)

# padding each one to ensure same max length

padded_tokenized_ground_train = tensorflow.keras.utils.pad_sequences(tokenized_ground_train, padding="post", maxlen=200)
padded_tokenized_ground_test = tensorflow.keras.utils.pad_sequences(tokenized_ground_test, padding="post", maxlen=200)
padded_tokenized_false_train = tensorflow.keras.utils.pad_sequences(tokenized_false_train, padding="post", maxlen=200)
padded_tokenized_false_test = tensorflow.keras.utils.pad_sequences(tokenized_false_test, padding="post", maxlen=200)

# these lists will store our values to plot

ground_train_accuracy_vs_depth = list()
ground_test_accuracy_vs_depth = list()
false_train_accuracy_vs_depth = list()
false_test_accuracy_vs_depth = list()


for i in range(1,30):
    ground_tree = DecisionTreeClassifier(max_depth=i).fit(padded_tokenized_ground_train, ground_train_y)
    false_tree = DecisionTreeClassifier(max_depth=i).fit(padded_tokenized_false_train, false_train_y)

    ground_training_prediction = ground_tree.predict(padded_tokenized_ground_train)
    ground_testing_prediction = ground_tree.predict(padded_tokenized_ground_test)

    false_training_prediction = false_tree.predict(padded_tokenized_false_train)
    false_testing_prediction = false_tree.predict(padded_tokenized_false_test)

    ground_training_accuracy = sk.metrics.accuracy_score(ground_train_y, ground_training_prediction, normalize=True)
    ground_testing_accuracy = sk.metrics.accuracy_score(ground_test_y, ground_testing_prediction)

    false_training_accuracy = sk.metrics.accuracy_score(false_train_y, false_training_prediction, normalize=True)
    false_testing_accuracy = sk.metrics.accuracy_score(false_test_y, false_testing_prediction)
    print("\n\tDepth: {}\n\n".format(i))
    ground_train_accuracy_vs_depth.append(ground_training_accuracy)
    ground_test_accuracy_vs_depth.append(ground_testing_accuracy)
    false_train_accuracy_vs_depth.append(false_training_accuracy)
    false_test_accuracy_vs_depth.append(false_testing_accuracy)

    print("Accuracy for ground truth (training vs testing): {} vs {}\nAccuracy for false truth (training vs testing): {} vs {}".format(ground_training_accuracy, ground_testing_accuracy, false_training_accuracy, false_testing_accuracy))

indexs = list([i for i in range(1,30)])
plt.plot(indexs, ground_train_accuracy_vs_depth, label="Ground_train_accuracy")
plt.plot(indexs,ground_test_accuracy_vs_depth, label="Ground_test_accuracy")
plt.plot(indexs,false_train_accuracy_vs_depth, label="False_train_accuracy")
plt.plot(indexs,false_test_accuracy_vs_depth, label="False_test_accuracy")
plt.legend()
plt.xlabel("Max depth of decision tree")
plt.ylabel("Accuracy of prediction")
plt.title("Accuracy of Decision Tree's vs respective Max Depth")
plt.show()