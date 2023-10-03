import pandas as pd

# df = pd.read_csv("../IMDB_with_predictions.csv")
adjectives = pd.read_csv("Adjectives.csv").Word.astype(str)

# print(adjectives)
for i in range(len(adjectives)):
    if(adjectives[i].strip() == "a"):
        print("Here")

print(adjectives.iloc[1].strip())