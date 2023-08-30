import tkinter as tk
from tkinter import *
from tkinter import ttk
from Tooltip import *

# Node class to assist with the creation of the visualisation
class Node:
    """Node class to assist with creation of visualisation
    Node constructor takes x coordinate, y coordinate, left child, right child, text and prediction by NN
    Node class has following public methods:
        get_pos_x()
        get_pos_y()
        get_LC()
        get_RC()
        get_text()
        get_prediction()
        set_LC()
        set_RC()
        draw_to_scrn(window)
            -> Takes a tkinter window or widget to place the current node in
            returns none
    pretty intuitive
    """
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
    
    def draw_to_scrn(self, window, subset_char_length):
        ttk.Label(window, text=self.prediction, width=20, wraplength=120, justify="center", font=("Arial", 8) ).grid(column=int(self.x), row=self.y)

        if(len(self.text) < subset_char_length):
            ttk.Label(window, text=self.text, width=20,  wraplength=120, justify="center", font=("Arial", 8)).grid(column=int(self.x), row=self.y+1)
        else:
            words = self.text[0:subset_char_length:1]
            words += "..."
            para = ttk.Label(window, text=words, width=20,  wraplength=120, justify="center", font=("Arial", 8))
            para.grid(column=int(self.x), row=self.y+1)
            new = ToolTip(para, self.text)

    def draw_root_to_scrn(self, window, subset_char_length):
        ttk.Label(window, text=self.prediction, width=30, wraplength=150, justify="center", font=("Arial", 8) ).grid(column=int(self.x), row=self.y)

        if(len(self.text) < subset_char_length):
            para = ttk.Label(window, text=self.text, width=30,  wraplength=150, justify="center", font=("Arial", 8)).grid(column=int(self.x), row=self.y+1)

        else:
            words = self.text[0:subset_char_length:1]
            words += "..."
            para = ttk.Label(window, text=words, width=30,  wraplength=150, justify="center", font=("Arial", 8))
            para.grid(column=int(self.x), row=self.y+1)
            new = ToolTip(para, self.text)

def preorder_traversal_text(node):
    if(node == None): return
    
    print("Parent: {}".format(node.get_text()))
    if node.get_LC() != None:
        print("L_Child: {}".format(node.get_LC().get_text()))
    if node.get_RC() != None:
        print("R_Child: {}".format(node.get_RC().get_text()))
    
    preorder_traversal_text(node.get_LC())
    preorder_traversal_text(node.get_RC())

    return

def preorder_traversal_co_ords(node):
    if(node == None): return
    
    print("Parent: ({},{})".format(node.get_pos_x(), node.get_pos_y()))
    if node.get_LC() != None:
        print("L_Child: ({},{})".format(node.get_LC().get_pos_x(), node.get_LC().get_pos_y()))
    if node.get_RC() != None:
        print("R_Child: ({},{})".format(node.get_RC().get_pos_x(), node.get_RC().get_pos_y()))
    
    preorder_traversal_co_ords(node.get_LC())
    preorder_traversal_co_ords(node.get_RC())

    return

