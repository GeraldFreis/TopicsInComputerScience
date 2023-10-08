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


def stringify(sentence_list: list, lower_bound: int, upper_bound: int, delim)->str:
    """Function to convert sentence list into strings"""
    sentences_in_bounds = [sentence_list[i] for i in range(lower_bound, upper_bound)]
    sub_para = str()
    for sentence in sentences_in_bounds:
        sub_para += sentence + delim
    return sub_para


def Splitting_texts(paragraph: str, max_depth: int, current_layer: int, main_list: list, delim)->list:
    """
    Splitting_texts takes a paragraph and a max_depth, current layer and main list and delimeter to split by and returns a list of CNN's prediction for each recursive subsection of the paragraph
        - for sentence level splitting delim=" "
        - for paragraph level splitting delim="."
    """
    # splitting sentences and removing elipses and malfunctions
    sentences = paragraph.split(delim)
    sentences = [sentence for sentence in sentences if len(sentence) >= 2]

    if(len(sentences) <= 1 or max_depth==current_layer): 
        main_list.append({"Layer": current_layer, "A": ".", "Prediction_A": "a", "B": ".", "Prediction_B": "b"})
        return main_list

    # getting the subsets of the paragraph    
    A = stringify(sentences, 0, int(len(sentences)/2), delim)
    B = stringify(sentences, int(len(sentences)/2), len(sentences), delim)

   
    layer = {"Layer": current_layer, "A": A, "Prediction_A": "padded_A", "B": B, "Prediction_B": "padded_B"}
    main_list.append(layer)

    # print("Parent: {}\nLeft: {}\n Right: {}\n\n".format(paragraph, A, B))

    # recursively getting the next layers
    main_list = Splitting_texts(A, max_depth, current_layer+1, main_list, delim)
    main_list = Splitting_texts(B, max_depth, current_layer+1, main_list, delim)
    
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


    predictions_list = model.predict(total_prediction_list, verbose=False)

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

def simpler_drawing(Root, tree_list, window, max_depth, wrap_length):
    """
    simpler_drawing takes a list of the tree, the window, and  max depth, wrap_length
    The function draws to screen where a nodes position is given by previous_node_on_layer+ 2^(n-l)
        - Where n is the max depth, and l is the current layer
        - previous_node_on_layer is 0 for first node and then 2^(n-l) for the second etc.
    """
    subset_char_length_to_display = 400
    Root.draw_root_to_scrn(window, subset_char_length_to_display+100, wrap_length+30)

    for i in range(1, max_depth):
        # getting all of the layers with the current list
        last_pos = 0
        current_layer_list = list([layer for layer in tree_list if layer.get("Layer") == i])

        for current_val in current_layer_list: # for each layer we want to add the parts of that layer to the right space
            last_pos += pow(2, (max_depth-i-1)) # adding our current x axis increment which is explained above
            if (current_val.get("A") != "."):
                node = Node(last_pos, i*3, None, None, current_val.get("A"), current_val.get("Prediction_A")) # initialising the node
                node.draw_to_scrn(window, subset_char_length_to_display-200, wrap_length) # drawing to window
            else:
                node = Node(last_pos, i*3, None, None, " ", " ") # initialising the node
                node.draw_to_scrn(window, subset_char_length_to_display-200, wrap_length) # drawing to window

            last_pos += pow(2, (max_depth-i))
            if (current_val.get("B") != "."):
                newnode = Node(last_pos, i*3, None, None, current_val.get("B"), current_val.get("Prediction_B"))
                newnode.draw_to_scrn(window, subset_char_length_to_display-200, wrap_length)
            else:
                newnode = Node(last_pos, i*3, None, None, " ", " ")
                newnode.draw_to_scrn(window, subset_char_length_to_display-200, wrap_length)

            last_pos += pow(2, (max_depth-i-1)) # ensuring that we have ample space to the nodes to our current right
