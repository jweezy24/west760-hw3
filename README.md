# Homework 3 Repository
This repo contains two files. The `main.py` file contains the majority of code.
This document will be a guide as to how to generate all of the data within homework 3.

## Dependencies

To install dependcies you do the usual, `pip install -r requirements.txt`.

## How to run

Execute `python3 main.py` to generate all figures and information the programming section of the homework.
If you would like to disable or enable figures to generate, you have to comment/uncomment lines of code within the main.
More information reguarding the process can be found in the notes section.

To generate the ROC plot for one of the questions in the earlier section, execute `python3 roc_plot.py`.

## Notes
 
There is both a implementation for nearest neighbors from scratch and I also use the sklearn implementation.
The reason for this is to verify if I was getting the correct predict probabilities.

The code can take a while to run.
I designed the code to executed in parts so that you may have the easiest time evaluating each piece.
You will find that each question has labled sections in the main file.
Comment/uncomment each part to evaluate each as needed.

For question 3, to save time, I hardcoded the results.
Each fold evaluation takes about 25 seconds, thus to not waste your time, I have hard coded the results from the functions.

For question 5, I used the nearest neighbor library for the confidence values.
The main reason for this is because the results do not mimick the example ROC plot in the homework file.
I didn't believe it and didn't trust my own code so I used the confidence values from sklearn. 