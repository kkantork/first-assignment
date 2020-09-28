#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:25:40 2020

@author: rotnak
"""

import csv
import pandas as pd
import numpy as np

df = pd.read_csv("./train.tsv", sep='\t', names=['cena', 'liczba', 'm2', 'ad', 'region', 'opis'])
df2 = pd.read_csv("./description.csv", sep=',')#, skiprows=[0])

print(df)

#print(df.reset_index(drop=True) == df2.reset_index(drop=True))
#df.loc[df.iloc[:,1] == df2.iloc[:,0], df.iloc[:,6]] = 'OK'

frames = [df, df2]
res1 = pd.concat([df, df2], axis=1)
print(res1)



