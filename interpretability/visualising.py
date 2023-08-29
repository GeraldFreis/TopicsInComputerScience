import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tkinter as tk
from tkinter import *
from tkinter import ttk
import math as m
import numpy as np
from node import *
from SplittingPredictions import *

CNN = keras.models.load_model("../CNN_Non_Dense")
raw_data = pd.read_csv("IMDB.csv")




def readingInTree(window, maxsize, sub_intervals, data_frame, index, model):
    tree_list = Splitting_Predictions(data_frame.iloc[index].review, sub_intervals, 1, list(), model)
    Root = Node(int(maxsize/2), 0, None, None, text=data_frame.iloc[index].review, prediction=data_frame.iloc[index].sentiment)
    Drawing_nodes_to_screen(Root, 0, tree_list, window, sub_intervals)
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