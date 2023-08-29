import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer

CNN = keras.models.load_model("../CNN_Non_Dense")
raw_data = pd.read_csv("IMDB.csv")
first_paragraph = raw_data.review[0]

"""Function to automate tokenizing and padding"""
def tokenization(string_to_padd_and_tok: str, tokenizer):
    tokenized = tokenizer.texts_to_sequences(string_to_padd_and_tok)
    padded = tf.keras.utils.pad_sequences(tokenized, maxlen=1000, padding="post")
    return padded

"""Function to convert sentence list into strings"""
def stringify(sentence_list: list, lower_bound: int, upper_bound: int)->str:
    sentences_in_bounds = [sentence_list[i] for i in range(lower_bound, upper_bound)]
    sub_para = str()
    for sentence in sentences_in_bounds:
        sub_para += sentence + "."
    return sub_para

"""
Tree_Creator takes a paragraph and a max_depth and returns a list of CNN's prediction for each recursive subsection of the paragraph
"""
def Tree_Creator(paragraph: str, max_depth: int, current_layer: int, main_list: list)->list:
    # splitting sentences and removing elipses and malfunctions
    sentences = paragraph.split(".")
    sentences = [sentence for sentence in sentences if len(sentence) >= 2]

    if(len(sentences) <= 1 or max_depth==current_layer): return main_list

    # getting the subsets of the paragraph    
    A = stringify(sentences, 0, int(len(sentences)/2))
    B = stringify(sentences, int(len(sentences)/2), len(sentences))

    tokenizer = Tokenizer(num_words=max(len(A), len(B)))
    tokenizer.fit_on_texts(A)

    # padding the texts and tokenizing
    padded_A = tokenization(A, tokenizer)
    padded_B = tokenization(B, tokenizer)

    # loading the CNN
    model = keras.models.load_model("../CNN_Non_Dense")

    # running the strings through the CNN and getting the prediction
    prediction_A = model.predict(padded_A, verbose=1)
    prediction_B = model.predict(padded_B, verbose=1)

    avg_A = float(sum(prediction_A)) / float(len(prediction_A))
    avg_B = float(sum(prediction_B)) / float(len(prediction_B)) 

    layer = {"Layer": current_layer, "A": A, "Prediction_A": avg_A, "B": B, "Prediction_B": avg_B}
    main_list.append(layer)

    # recursively getting the next layers
    main_list = Tree_Creator(A, max_depth, current_layer+1, main_list)
    main_list = Tree_Creator(B, max_depth, current_layer+1, main_list)
    
    return main_list

print(Tree_Creator(first_paragraph, 2, 1, list()))
