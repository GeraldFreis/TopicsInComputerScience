import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tkinter as tk

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


tree = Tree_Creator(first_paragraph, 5, 1, list())
from tkinter import *
from tkinter import ttk

def TreeVisualiser(layer_list: list, root_str: str, sub_intervals: int)->None:
    root = tk.Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()  

    my_scrollbar = tk.Scrollbar(frm, orient='horizontal')
    canvas = tk.Text(frm, xscrollcommand=my_scrollbar.set)
    my_scrollbar.pack(side=BOTTOM, fill=X)
    # canvas['xscrollcommand'] = my_scrollbar.set

    center = 7*sub_intervals
    ttk.Label(canvas, text="1", font=("Arial", 8)).grid(column=center, row=0)
    ttk.Label(canvas, text=root_str, width=50, wraplength=400, justify="center", font=("Arial", 8)).grid(column=center, row=1)

    row = 2
    for i in range(1, sub_intervals):
        LHS_col = center - i - 2
        RHS_col = center + i + 2

        indexs = [layer for layer in range(len(layer_list)) if layer_list[layer].get("Layer") == i]
        for ind in indexs:
            ttk.Label(canvas, text=layer_list[ind].get("Prediction_A"), width=50, wraplength=400, justify="center").grid(column=LHS_col, row=row)
            ttk.Label(canvas, text=layer_list[ind].get("A"), width=50, wraplength=400, justify="center").grid(column=LHS_col, row=row+1)

            ttk.Label(canvas, text=layer_list[ind].get("Prediction_B"), width=50, wraplength=400, justify="center").grid(column=RHS_col, row=row)
            ttk.Label(canvas, text=layer_list[ind].get("B"), width=50, wraplength=400, justify="center").grid(column=RHS_col, row=row+1)

            LHS_col += 2
            RHS_col -= 2

        row += 2
    canvas.pack(side=TOP, fill=X)
    my_scrollbar.config(command=canvas.xview)

    # canvas.pack(side=TOP, fill=BOTH) 
    root.mainloop()

TreeVisualiser(tree, first_paragraph, 4)