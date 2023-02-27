# Homework 3 Repository
This repo contains two files. The `main.py` file contains the majority of code.
This document will be a guide as to how to generate all of the data within homework 3.

## Dependencies

To install dependcies you do the usual, `pip install -r requirements.txt`.

## How to run

Execute `python3 main.py` to generate all figures and information the programming section of the homework.
To generate the ROC plot for one of the questions in the earlier section, execute `python3 roc_plot.py`.

## Notes
 
There is both a implementation for nearest neighbors from scratch and I also use the sklearn implementation.
The reason for this is to verify if I was getting the correct predict probabilities.

The code can take a while to run.
I designed the code to executed in parts so that you may have the easiest time evaluating each piece.
You will find that each question has labled sections in the main file.
Comment/uncomment each part to evaluate each as needed.
