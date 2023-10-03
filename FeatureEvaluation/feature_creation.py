import pandas as pd

df = pd.read_csv("../IMDB_with_predictions.csv")
adjectives = pd.read_csv("Adjectives.csv")
adjectives = adjectives.Word.astype(str)


# features will be number of adjectives / proportion of adjectives
# number of nouns / proportion of nouns
proportion_of_adjectives = list()
for i in range(len(df)):
    print(i)
    words = df["review"].iloc[i].split(" ")
    adjectives_count = 0
    overall_words = 0
    # print(words)
    for word in words:
        # newword = " " + word
        for adjective in adjectives:
            if(word == adjective.strip()):
                adjectives_count += 1
                # print("works")
                break;
        overall_words += 1
    
    proportion_of_adjectives.append(float(float(adjectives_count) / float(overall_words)))
    # print(adjectives_count)

df["proportion_of_adjectives"] = proportion_of_adjectives
df.to_csv("IMDB_with_features.csv", index=False)