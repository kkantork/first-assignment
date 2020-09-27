#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:37:55 2020

@author: rotnak
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from scipy import stats

df = pd.read_csv('survey_results_public.csv')
schema_df = pd.read_csv('survey_results_schema.csv')

#df.info()

pd.set_option('display.max_columns', 85)

df_reg = pd.read_csv('survey_results_public.csv', usecols = ['Age', 'YearsCode', 'YearsCodePro', 'Employment', 'Hobbyist'])

df_reg.dropna(inplace = True)

column_values = df_reg[['YearsCodePro']].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)

column_values = df_reg[['YearsCode']].values
unique_values = np.unique(column_values)
print(unique_values)

df_reg.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                  inplace=True)

df_reg = df_reg.astype({'YearsCodePro': 'int64', 'YearsCode': 'int64'} , copy=False)
#df_reg = df_reg.astype('float64', copy=False)
print(df_reg['YearsCode'])
print(df_reg['YearsCodePro'])


print(df_reg.corr())

plt.plot(df_reg['YearsCodePro'], df_reg['Age'],'ro', markersize=0.3)
plt.ylabel('YearsCodePro')
plt.xlabel('Age')
plt.plot(df_reg['YearsCodePro'], df_reg['YearsCode'],'ro', markersize=0.3, color = 'blue')
plt.ylabel('YearsCodePro')
plt.xlabel('YearsCode/Age')
plt.show()

print(df_reg.dtypes)

