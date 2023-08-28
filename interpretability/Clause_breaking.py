import pandas as pd
import numpy as np

# removing all <br> from the paragraphs and turning each sentence into a separate test
raw = pd.read_csv("IMDB.csv")
sentences_list = list()
corresponding_CNN_evaluations = list()
corresponding_ground_evaluations = list()
related_para_ID = list()

for i in range(len(raw["review"])):
    print(i)
    sentences = raw.iloc[i].review.split(".")
    try:
        sentences.remove("")
    except Exception:
        continue;
    
    for sentence in sentences:
        if(len(sentence) > 2):
            sentences_list.append(sentence)
            corresponding_CNN_evaluations.append(raw.iloc[i].CNN_Predictions)
            corresponding_ground_evaluations.append(raw.iloc[i].sentiment)
            related_para_ID.append(i)

print(len(sentences_list))
print("\nFinished with deconstructing paragraphs\n\n")
newDf = pd.DataFrame(columns=["Related_paragraph_ID", "Sentences", "CNN_paragraph_prediction", "Ground_paragraph_prediction"])# data=[sentences_list, corresponding_CNN_evaluations, corresponding_ground_evaluations])
newDf["Sentences"] = sentences_list
newDf["CNN_paragraph_prediction"] = corresponding_CNN_evaluations
newDf["Ground_paragraph_prediction"] = corresponding_ground_evaluations
newDf["Related_paragraph_ID"] = related_para_ID

newDf["Sentences"].dropna(inplace=True)
print(newDf)
newDf.to_csv("IMDB_sentences_broken.csv", index=False)