from visualising_paragraphs import *
from visualising_sentences import *
import keras 
import pandas

# reading in our DF and CNN
CNN = keras.models.load_model("../CNN_Non_Dense/")
paragraphs = pandas.read_csv("IMDB.csv")
sentences = pandas.read_csv("IMDB_sentences.csv")


TreeVisualiser_sentences(4, sentences, 0, CNN)
TreeVisualiser_paragraphs(5, paragraphs, 0, CNN)