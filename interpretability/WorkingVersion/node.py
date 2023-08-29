import tkinter as tk
from tkinter import *
from tkinter import ttk

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
    
    def draw_to_scrn(self, window):
        ttk.Label(window, text=self.prediction, width=20, wraplength=120, justify="center", font=("Arial", 8) ).grid(column=int(self.x), row=self.y)

        if(len(self.text) < 100):
            ttk.Label(window, text=self.text, width=20,  wraplength=120, justify="center", font=("Arial", 8)).grid(column=int(self.x), row=self.y+1)
        else:
            words = self.text[0:100:1]
            words += "..."
            ttk.Label(window, text=words, width=20,  wraplength=120, justify="center", font=("Arial", 8)).grid(column=int(self.x), row=self.y+1)


