import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer

CNN = keras.models.load_model("../CNN_Non_Dense")
raw_data = pd.read_csv("IMDB.csv")
first_paragraph = raw_data.review[0]
"""
Tree_Creator takes a paragraph and a max_depth and returns a list of CNN's prediction for each recursive subsection of the paragraph
"""
def Tree_Creator(paragraph: str, max_depth: int, current_layer: int, main_list: list)->list:
    # splitting sentences and removing elipses and malfunctions
    sentences = paragraph.split(".")
    sentences = [sentence for sentence in sentences if len(sentence) >= 2]

    if(len(sentences) <= 1 or max_depth==current_layer): return main_list

    # computing each layer of sentences
    A = [sentences[i] for i in range(0, int(len(sentences)/2))]
    B = [sentences[i] for i in range(int(len(sentences)/2), len(sentences))]

    string_A = str()
    string_B = str()

    for sentence in A:
        string_A += sentence + "."
    for sentence in B:
        string_B += sentence + "."

    tokenizer = Tokenizer(num_words=max(len(string_A), len(string_B)))
    tokenizer.fit_on_texts(string_A)
    tokenized_A = tokenizer.texts_to_sequences(string_A)
    tokenized_B = tokenizer.texts_to_sequences(string_B)

    # padding the texts
    padded_A = tf.keras.utils.pad_sequences(tokenized_A, maxlen=1000, padding="post")
    padded_B = tf.keras.utils.pad_sequences(tokenized_B, maxlen=1000, padding="post")

    # loading the CNN
    model = keras.models.load_model("../CNN_Non_Dense")

    # running the strings through the CNN and getting the prediction
    prediction_A = model.predict(padded_A, verbose=1)
    prediction_B = model.predict(padded_B, verbose=1)

    avg_A = float(sum(prediction_A)) / float(len(prediction_A))
    avg_B = float(sum(prediction_B)) / float(len(prediction_B)) 

    layer = {"Layer": current_layer, "A": string_A, "Prediction_A": avg_A, "B": string_B, "Prediction_B": avg_B}
    main_list.append(layer)

    main_list = Tree_Creator(string_A, max_depth, current_layer+1, main_list)
    main_list = Tree_Creator(string_B, max_depth, current_layer+1, main_list)
    
    return main_list

print(Tree_Creator(first_paragraph, 7, 1, list()))
