#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:59:49 2020

@author: rotnak
"""
import csv
import pandas as pd
import numpy as np

df = pd.read_csv("./train.tsv", sep='\t')
df1 = str(round(df.iloc[:, 0].mean(),0)# f = open("out0.csv", "x")

# with open("out0.csv", 'w', newline='') as f:
#     f.write(df1)
)

# f = open("out0.csv", "x")

# with open("out0.csv", 'w', newline='') as f:
#     f.write(df1)

#adding a new column for price per m2 calculation
df[len(df.columns)] = 6

#price per m2 calc
df.iloc[:,6] = df.iloc[:, 0] / df.iloc[:, 2]

#srednia cena za metr2
avg = df.iloc[:,6].mean()
print(avg)

df_col = df.iloc[:,[0, 1, 6]]
#print(df_col)
df_rows = df_col.loc[df.iloc[:,1]>=3]
print(df_rows)
df_rows2 = df_rows.loc[df.iloc[:,6]<avg]
print(df_rows2)

df_rows2.to_csv('out1.csv', sep='\t', index=False, float_format='%.3f')

#df.to_csv('proba2.csv',columns=df.iloc[:,[0, 1, 6]], sep='\t', index=False, float_format='%.3f')



# print(df)