"""
visualising_sentences.py has two functions: The most important is TreeVisualiser_sentences(depth, data_frame, index, model)
    - The function mentioned produces a tree from a given data entry in the dataframe
"""

import pandas as pd
import keras
import tensorflow as tf
import tkinter as tk
from tkinter import *
import tkinter.ttk as ttk

import math as m
import numpy as np
from node import *
from SplittingPredictions import *

CNN = keras.models.load_model("../CNN_Non_Dense")
raw_data = pd.read_csv("IMDB_sentences.csv")

def Composing_Tree(window, maxsize, sub_intervals, data_frame, index, model):
    """Function that generates the split predictions for each paragraph, and then composes a tree from the split predictions"""
    wrap_length = 100 # for sentences wrap length of 100 seems to work best
    tree_list = Splitting_texts(data_frame.iloc[index].Sentences, sub_intervals, 1, list(), delim=" ") # generating our list of trees 
    tree_list = predictions(tree_list, model)
    Root = Node(int(maxsize/2), 0, None, None, text=data_frame.iloc[index].Sentences, prediction=data_frame.iloc[index].Sentence_CNN_prediction) # getting our tree set up with a root node
    # Drawing_nodes_to_screen(Root, 0, tree_list, window, sub_intervals) # setting up our tree structure with our root node and drawing the nodes to screen 
    simpler_drawing(Root, tree_list, window, sub_intervals, wrap_length)

    return Root


def TreeVisualiser_sentences(depth: int, data_frame, index, model)->None:
    """Function that takes in the root node and sub_interval limits and visualises the tree"""

    max_size = pow(2, depth)

    # setting screen up
    root = tk.Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    frm.rowconfigure(max_size*3) 
    frm.columnconfigure(max_size)

    # setting scrollbar up
    my_scrollbar = tk.Scrollbar(frm, orient='horizontal')
    canvas = tk.Canvas(frm, xscrollcommand=my_scrollbar.set)

    my_scrollbar.pack(side=BOTTOM, fill=X)
    # canvas['xscrollcommand'] = my_scrollbar.set

    Root = Composing_Tree(canvas, max_size, depth, data_frame, index, model) # creating our tree with our predictions, drawing our tree to screen
    # preorder_traversal_co_ords(node=Root)
    
    canvas.pack(side=TOP, fill=X)
    my_scrollbar.config(command=canvas.xview)
    
    root.mainloop()
    return

# TreeVisualiser_sentences(depth=5, data_frame=raw_data, index=0, model=CNN)
