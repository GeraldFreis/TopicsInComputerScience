import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tkinter as tk
from tkinter import *
from tkinter import ttk
import math as m
import numpy as np

CNN = keras.models.load_model("../CNN_Non_Dense")
raw_data = pd.read_csv("IMDB.csv")

"""Function to automate tokenizing and padding"""
def tokenization(string_to_padd_and_tok: str, tokenizer):
    tokenized = tokenizer.texts_to_sequences(string_to_padd_and_tok)
    padded = tf.keras.utils.pad_sequences(tokenized, maxlen=1000, padding="post")
    padded = (padded)
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
def Tree_Creator(paragraph: str, max_depth: int, current_layer: int, main_list: list, model)->list:
    # splitting sentences and removing elipses and malfunctions
    sentences = paragraph.split(".")
    sentences = [sentence for sentence in sentences if len(sentence) >= 2]

    if(len(sentences) <= 1 or max_depth==current_layer): return main_list

    # getting the subsets of the paragraph    
    A = stringify(sentences, 0, int(len(sentences)/2))
    B = stringify(sentences, int(len(sentences)/2), len(sentences))

    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(A)
    tokenizer.fit_on_texts(B)

    # padding the texts and tokenizing

    padded_A = tokenization(A, tokenizer)
    padded_B = tokenization(B, tokenizer)

    # running the strings through the CNN and getting the prediction
    prediction_A = model.predict(padded_A, verbose=0)

    prediction_B = model.predict(padded_B, verbose=0)
    
    prediction_A_occurrences = dict()
    for i in range(len(prediction_A)):
        if prediction_A[i][0] not in prediction_A_occurrences:
            prediction_A_occurrences[prediction_A[i][0]] = 1
        else:
            prediction_A_occurrences[prediction_A[i][0]] += 1
    
    prediction_B_occurrences = dict()
    for i in range(len(prediction_B)):
        if prediction_B[i][0] not in prediction_B_occurrences:
            prediction_B_occurrences[prediction_B[i][0]] = 1
        else:
            prediction_B_occurrences[prediction_B[i][0]] += 1
    
    avg_A = float()
    for i in range(len(prediction_A)):
       avg_A += prediction_A[i][0]*(prediction_A_occurrences[prediction_A[i][0]])
    avg_B = float()
    for i in range(len(prediction_B)):
        avg_B += prediction_B[i][0]*(prediction_B_occurrences[prediction_B[i][0]])

    avg_A /= len(prediction_A)*100
    avg_B /= len(prediction_B)*100
    # avg_A = float(sum(prediction_A)) / float(len(prediction_A))
    # avg_B = float(sum(prediction_B)) / float(len(prediction_B)) 

    layer = {"Layer": current_layer, "A": A, "Prediction_A": avg_A, "B": B, "Prediction_B": avg_B}
    main_list.append(layer)

    # recursively getting the next layers
    main_list = Tree_Creator(A, max_depth, current_layer+1, main_list, model)
    main_list = Tree_Creator(B, max_depth, current_layer+1, main_list, model)
    
    return main_list

# tree = Tree_Creator(first_paragraph, 6, 1, list())

# Class to aid with the visualisation
class Node:
    def __init__(self, x, y, child_left, child_right, text, prediction):
        self.x = x
        self.y = y
        self.child_left = child_left
        self.child_right = child_right
        self.text = text
        self.prediction = prediction

    def get_pos_x(self):
        return self.x
    
    def get_pos_y(self):
        return self.y
    
    def get_LC(self):
        return self.child_left
    
    def get_RC(self):
        return self.child_right
    
    def get_text(self):
        return self.text
    
    def get_prediction(self):
        return self.prediction
    
    def set_LC(self, LC):
        self.child_left = LC
        return

    def set_RC(self, RC):
        self.child_right = RC
        return
    
    def draw_to_scrn(self, window):
        ttk.Label(window, text=self.prediction, width=20, wraplength=400, justify="center", font=("Arial", 8) ).grid(column=int(self.x), row=self.y)
        ttk.Label(window, text=self.text, width=20,  wraplength=400, justify="center", font=("Arial", 8)).grid(column=int(self.x), row=self.y+1)


def tree_creator_with_nodes(Root, current_index, tree_list, window, sub_intervals):
    # creating the tree with nodes
    LC_position = 2*current_index + 1
    RC_position = 2*current_index + 2

    if(current_index >= len(tree_list)): return

    children = tree_list[current_index]

    L_C_N_position = float(Root.get_pos_x() - pow(2, (sub_intervals - children.get("Layer"))))
    R_C_N_position = float(m.ceil(Root.get_pos_x() + pow(2, (sub_intervals - children.get("Layer")))))

    Left_Child_Node = Node(L_C_N_position, int(Root.get_pos_y() + 3), None, None, children.get("A"), children.get("Prediction_A"))
    Right_Child_Node = Node(R_C_N_position, int(Root.get_pos_y() + 3), None, None, children.get("B"), children.get("Prediction_B"))

    Root.set_LC(Left_Child_Node)
    Root.set_RC(Right_Child_Node)

    Root.draw_to_scrn(window)
    tree_creator_with_nodes(Left_Child_Node, LC_position, tree_list, window, sub_intervals)
    tree_creator_with_nodes(Right_Child_Node, RC_position, tree_list, window, sub_intervals)

    return

    


def readingInTree(window, maxsize, sub_intervals, data_frame, index, model):
    tree_list = Tree_Creator(data_frame.iloc[index].review, sub_intervals, 1, list(), model)
    Root = Node(int(maxsize/2), 0, None, None, text=data_frame.iloc[index].review, prediction=data_frame.iloc[index].sentiment)
    tree_creator_with_nodes(Root, 0, tree_list, window, sub_intervals)
    return Root


def TreeVisualiser(sub_intervals: int, data_frame, index, model)->None:
    """Function that takes in the root node and sub_interval limits and visualises the tree"""

    max_size = pow(2, sub_intervals+1)

    root = tk.Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    frm.rowconfigure(max_size*3) 
    frm.columnconfigure(max_size)
    my_scrollbar = tk.Scrollbar(frm, orient='horizontal')
    canvas = tk.Text(frm, xscrollcommand=my_scrollbar.set)
    my_scrollbar.pack(side=BOTTOM, fill=X)
    # canvas['xscrollcommand'] = my_scrollbar.set
    readingInTree(canvas, max_size, sub_intervals, data_frame, index, model)
    canvas.pack(side=TOP, fill=X)
    my_scrollbar.config(command=canvas.xview)

    # canvas.pack(side=TOP, fill=BOTH) 
    root.mainloop()

TreeVisualiser(5, raw_data, 0, CNN)