# Gerald Freislich Topics in Computer Science

## Interesting Results
- A CNN trained on the IMDB_dataset_final.csv (50,000 rows) for sentiment analysis has a testing accuracy of c. 0.889
->SentimentNLP.py
- A DT trained on the IMDB_dataset_final.csv for sentiment analysis has a testing accuracy of c. 0.526
->SentimentDTGround.py
- But, a DT trained on the IMDB_with_predictions.csv (A DT trained to predict the predictions of the CNN) for sentiment analysis has a testing accuracy of c. 0.57
->SentimentDTFalse.py

- A DT was trained to predict the CNN's predictions and achieved a max accuracy of 0.57 with a depth of 2
-> InterpretabilityDT.py

- A DT was trained to predict, from each layer of the CNN, the final sentiment and scored a max accuracy of 0.576 for the first dense layer
- ->Taking Dt Over CNN

- A DT visualisation model was created to break up each paragraph and show the CNN across subsets of that paragraph hence providing an inference based approach
- Working Decision Visualiser

## Dependencies:
* Tensorflow 2.11.0 and Keras 2.11.0
* Python 3.8
* Scikit-Learn 1.3.0
* Pandas 2.0.3
