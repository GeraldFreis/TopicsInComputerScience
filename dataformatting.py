import pandas as pd

dataset = pd.read_csv("IMDB Dataset.csv", low_memory=False)

dataset.replace(to_replace="negative", value=0, inplace=True)
dataset.replace(to_replace="positive", value=1, inplace=True)
print(dataset)

dataset.to_csv("IMDB_dataset_final.csv")

