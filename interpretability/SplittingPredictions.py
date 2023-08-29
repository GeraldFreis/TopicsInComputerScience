import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from node import *
import math as m


def tokenization(string_to_padd_and_tok: str, tokenizer):
    """Function to automate tokenizing and padding"""
    tokenized = tokenizer.texts_to_sequences(string_to_padd_and_tok)
    padded = tf.keras.utils.pad_sequences(tokenized, maxlen=1000, padding="post")
    padded = (padded)
    return padded


def stringify(sentence_list: list, lower_bound: int, upper_bound: int)->str:
    """Function to convert sentence list into strings"""
    sentences_in_bounds = [sentence_list[i] for i in range(lower_bound, upper_bound)]
    sub_para = str()
    for sentence in sentences_in_bounds:
        sub_para += sentence + "."
    return sub_para


def Splitting_Predictions(paragraph: str, max_depth: int, current_layer: int, main_list: list, model)->list:
    """
    Splitting_Predictions takes a paragraph and a max_depth and returns a list of CNN's prediction for each recursive subsection of the paragraph
    """
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
    main_list = Splitting_Predictions(A, max_depth, current_layer+1, main_list, model)
    main_list = Splitting_Predictions(B, max_depth, current_layer+1, main_list, model)
    
    return main_list



def Drawing_nodes_to_screen(Root, current_index, tree_list, window, sub_intervals):
    """
    Function takes the root node, current index, list of splits, window and subintervals as parameters
    function returns none, but draws each node to the screen and initialises its child nodes
    """
    # creating the tree with nodes
    LC_position = 2*current_index + 1
    RC_position = 2*current_index + 2

    if(current_index >= len(tree_list)): return

    children = tree_list[current_index]
    # getting the position on the screen for the two children nodes
    L_C_N_position = float(Root.get_pos_x() - pow(2, (sub_intervals - children.get("Layer"))))
    R_C_N_position = float(m.ceil(Root.get_pos_x() + pow(2, (sub_intervals - children.get("Layer")))))

    # setting the children nodes up
    Left_Child_Node = Node(L_C_N_position, int(Root.get_pos_y() + 3), None, None, children.get("A"), children.get("Prediction_A"))
    Right_Child_Node = Node(R_C_N_position, int(Root.get_pos_y() + 3), None, None, children.get("B"), children.get("Prediction_B"))

    Root.set_LC(Left_Child_Node)
    Root.set_RC(Right_Child_Node)
    # drawing the root to screen
    Root.draw_to_scrn(window)
    # recursively drawing the children to screen in a postorder manner kinda
    Drawing_nodes_to_screen(Left_Child_Node, LC_position, tree_list, window, sub_intervals)
    Drawing_nodes_to_screen(Right_Child_Node, RC_position, tree_list, window, sub_intervals)
    return