#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:13:36 2020

@author: rotnak
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Selecting columns from survey
df_reg = pd.read_csv('survey_results_public.csv',
                     usecols=['Age', 'YearsCode', 'YearsCodePro'])

# Removing rows without values
df_reg.dropna(inplace=True)


# Verifying all values are numerical
column_values = df_reg[['YearsCodePro']].values.ravel()
unique_values = pd.unique(column_values)

column_values = df_reg[['YearsCode']].values
unique_values = np.unique(column_values)

# replacing string to int values
df_reg.replace(to_replace={'Less than 1 year': '0',
                           'More than 50 years': '51'},
               inplace=True)

# setting appropriate data types
df_reg = df_reg.astype({'YearsCodePro': 'int64', 'YearsCode': 'int64'},
                       copy=False)

# Correlation with outliers
print()
print("Correlation with outliers")
print(df_reg.corr())

# Boxplot with outliers
sns.boxplot(y='YearsCodePro', data=df_reg).set(
                              ylabel='YearsCodePro - outliers included')
plt.show()

# plot
plt.plot(df_reg['YearsCodePro'], df_reg['Age'], 'ro', markersize=0.3)
plt.ylabel('YearsCodePro')
plt.xlabel('Age')
plt.plot(df_reg['YearsCodePro'], df_reg['YearsCode'], 'ro',
         markersize=0.3, color='blue')
plt.ylabel('YearsCodePro - outliers included')
plt.xlabel('YearsCode/Age')
plt.show()


# interquartile range (IQR)/middle 50%

Q1 = df_reg.quantile(0.25)
Q3 = df_reg.quantile(0.75)
IQR = Q3 - Q1

df_reg_q = df_reg[~((df_reg < (Q1 - 1.5 * IQR)) |
                    (df_reg > (Q3 + 1.5 * IQR))).any(axis=1)]

# boxplot without outliers
sns.boxplot(y='YearsCodePro', data=df_reg_q).set(
                              ylabel='YearsCodePro - no outliers')
plt.show()

# plot
plt.plot(df_reg_q['YearsCodePro'], df_reg_q['Age'], 'ro', markersize=0.3)
plt.ylabel('YearsCodePro')
plt.xlabel('Age')
plt.plot(df_reg_q['YearsCodePro'], df_reg_q['YearsCode'], 'ro',
         markersize=0.3, color='blue')
plt.ylabel('YearsCodePro - no outliers')
plt.xlabel('YearsCode/Age')
plt.show()


print()
print("Correlation without outliers")
print(df_reg_q.corr())

# linear regression
reg2_q = linear_model.LinearRegression()
reg2_q.fit(df_reg_q[['YearsCode', 'Age']], df_reg_q['YearsCodePro'])

print()
print("coef:")
print(reg2_q.coef_)

print()
print("Predicted values for two variables without outliers:")
print(reg2_q.predict(df_reg_q[['YearsCode', 'Age']]))

y_pred = reg2_q.predict(df_reg_q[['YearsCode', 'Age']])
y_true = df_reg_q['YearsCodePro']

print()
print("MSE for two variables without outliers:")
print(mean_squared_error(y_true, y_pred))
