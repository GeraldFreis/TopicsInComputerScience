# measuring the impact / prediction based on the first sentence being positive
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("IMDB_sentences.csv")
indexs_covered = 0
first_sentences = list() # first sentence prediciton
corresponding_paragraph = list() # corresponding paragraph prediction
for i in range(len(data)):
    if(data.iloc[i].Related_paragraph_ID > indexs_covered):
        first_sentences.append(data.iloc[i].Sentence_CNN_prediction)
        corresponding_paragraph.append(data.iloc[i].CNN_paragraph_prediction)
        indexs_covered += 1

first_sentences = first_sentences[0: len(first_sentences): 1000]
corresponding_paragraph = corresponding_paragraph[0:len(corresponding_paragraph):1000]
# print(len(first_sentences))

plt.scatter(x=[i for i in range(0, len(first_sentences))],y=first_sentences, label="First sentence prediction")
plt.scatter(x=[i for i in range(0, len(corresponding_paragraph))], y=corresponding_paragraph, label="Total Paragraph Prediction")
plt.title("First Sentence Prediction vs Paragraph Total Prediction")
plt.legend()
plt.show()

