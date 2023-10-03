import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("IMDB_with_features.csv")
features = df.proportion_of_adjectives
sentiment_prediction = df.CNN_Predictions


train_x, test_x, train_y, test_y = train_test_split(features, sentiment_prediction, shuffle=True, random_state=1000, test_size=0.3)
DT = DecisionTreeClassifier().fit(train_x, train_y)
train_prediction = tree.predict(train_x)
test_prediction = tree.predict(test_x)

train_accuracy = sklearn.metrics.accuracy_score(train_prediction, train_y, normalize=True)
test_accuracy = sklearn.metrics.accuracy_score(test_prediction, test_y, normalize=True)

print("DT across training: {}\t DT across testing: {}\n".format(train_accuracy, test_accuracy))