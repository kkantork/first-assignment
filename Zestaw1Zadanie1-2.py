#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:59:49 2020

@author: rotnak
"""
import csv
import pandas as pd
import numpy as np

# read tsv file and calculate the price mean
df = pd.read_csv("./train.tsv", sep='\t')
df1 = str(round(df.iloc[:, 0].mean(), 0))

# write the output to file
with open("out0.csv", 'w', newline='') as f:
    f.write(df1)


# adding a new column for price per m2 calculation
df[len(df.columns)] = 6

# price per m2 calc - each row
df.iloc[:, 6] = df.iloc[:, 0] / df.iloc[:, 2]

# avg price per m2 according to col
avg = df.iloc[:, 6].mean()
print(avg)

# selecting cols with rooms, price, price per m2
df_col = df.iloc[:, [1, 0, 6]]

# filtering data with rooms >=3 and price < avg
df_rows = df_col.loc[df.iloc[:, 1] >= 3]
print(df_rows)
df_rows2 = df_rows.loc[df.iloc[:, 6] < avg]
print(df_rows2)

# print out the results to file
df_rows2.to_csv('out1.csv', sep='\t', index=False, float_format='%.3f')
