import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filename1 = "ztest1.csv"
filename2 = "ztest2.csv"
filename3 = "ztest3.csv"
filename4 = "ztest4.csv"
filename5 = "ztest5.csv"
filename6 = "ztest6.csv"
filename7 = "ztest7.csv"
filename8 = "ztest8.csv"
filename9 = "ztest9.csv"
filename10 = "ztest10.csv"


def read_csv(filepath):
    df = pd.read_csv(filepath, on_bad_lines="skip")
    df = df.dropna()
    return df

#mean_pos: mean values from the calibration test on the positive axis
#mean_neg: mean values from the calibration test on the negative axis
#g: theoretical value for g
offset = lambda mean_pos, mean_neg: (mean_pos - mean_neg)/2
scale_factor = lambda mean_pos, mean_neg, g: (mean_pos - mean_neg)/(2*g)


def main():

