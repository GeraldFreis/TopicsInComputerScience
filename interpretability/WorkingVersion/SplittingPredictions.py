import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from node import *
import math as m


def tokenization(string_to_padd_and_tok: list, tokenizer):
    """Function to automate tokenizing and padding"""
    tokenized = tokenizer.texts_to_sequences(string_to_padd_and_tok)
    padded = tf.keras.utils.pad_sequences(tokenized, padding="post", maxlen=1000)
    return padded


def stringify(sentence_list: list, lower_bound: int, upper_bound: int)->str:
    """Function to convert sentence list into strings"""
    sentences_in_bounds = [sentence_list[i] for i in range(lower_bound, upper_bound)]
    sub_para = str()
    for sentence in sentences_in_bounds:
        sub_para += sentence + "."
    return sub_para


def Splitting_texts(paragraph: str, max_depth: int, current_layer: int, main_list: list)->list:
    """
    Splitting_texts takes a paragraph and a max_depth and returns a list of CNN's prediction for each recursive subsection of the paragraph
    """
    # splitting sentences and removing elipses and malfunctions
    sentences = paragraph.split(".")
    sentences = [sentence for sentence in sentences if len(sentence) >= 2]

    if(len(sentences) <= 1 or max_depth==current_layer): return main_list

    # getting the subsets of the paragraph    
    A = stringify(sentences, 0, int(len(sentences)/2))
    B = stringify(sentences, int(len(sentences)/2), len(sentences))

   
    layer = {"Layer": current_layer, "A": A, "Prediction_A": "padded_A", "B": B, "Prediction_B": "padded_B"}
    main_list.append(layer)

    # print("Parent: {}\nLeft: {}\n Right: {}\n\n".format(paragraph, A, B))

    # recursively getting the next layers
    main_list = Splitting_texts(A, max_depth, current_layer+1, main_list)
    main_list = Splitting_texts(B, max_depth, current_layer+1, main_list)
    
    return main_list

def predictions(layer_list: list, model)->list:
    """Takes a list of layers (dict) returns the same layer list with predictions instead of padded values"""
    total_prediction_list = list()
    for layer in layer_list:
        predict_A = layer.get("A")
        predict_B = layer.get("B")
        total_prediction_list.append(predict_A)
        total_prediction_list.append(predict_B)
    
    # tokenizing and padding all of the prediction list
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(total_prediction_list)
    total_prediction_list = tokenization(total_prediction_list, tokenizer)


    predictions_list = model.predict(total_prediction_list)

    counter = 0
    for layer in layer_list:
        layer["Prediction_A"] = predictions_list[counter]
        layer["Prediction_B"] = predictions_list[counter+1]
        counter += 2
    
    return layer_list



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
    L_C_N_position = float(Root.get_pos_x() - pow(2, (sub_intervals - children.get("Layer")-1)))
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

def simpler_drawing(Root, tree_list, window, max_depth):
    """
    simpler_drawing takes a list of the tree, the window, and  max depth
    The function draws to screen where a nodes position is given by previous_node_on_layer+ 2^(n-l)
        - Where n is the max depth, and l is the current layer
        - previous_node_on_layer is 0 for first node and then 2^(n-l) for the second etc.
    """
    Root.draw_to_scrn(window)

    for i in range(1, max_depth):
        # getting all of the layers with the current list
        last_pos = 0
        current_layer_list = list([layer for layer in tree_list if layer.get("Layer") == i])

        for current_val in current_layer_list:
            print("\n\nLayer {}".format(i))
            last_pos += pow(2, (max_depth-i-1))
            node = Node(last_pos, i*3, None, None, current_val.get("A"), current_val.get("Prediction_A"))
            node.draw_to_scrn(window)

            print("Position of A: {}".format(last_pos))
            last_pos += pow(2, (max_depth-i))
            newnode = Node(last_pos, i*3, None, None, current_val.get("B"), current_val.get("Prediction_B"))
            newnode.draw_to_scrn(window)
            print("Position of B: {}".format(last_pos))
            last_pos += pow(2, (max_depth-i-1))
            print("Layer: {}, depth_adjustment: {}".format(i, pow(2, (max_depth-i-1))))
