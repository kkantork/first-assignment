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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error

df = pd.read_csv('survey_results_public.csv')
schema_df = pd.read_csv('survey_results_schema.csv')

#df.info()

pd.set_option('display.max_columns', 85)

#Selecting columns from survey
df_reg = pd.read_csv('survey_results_public.csv', usecols = ['Age', 'YearsCode', 'YearsCodePro'])
df_reg2 = pd.read_csv('survey_results_public.csv', usecols = ['Age', 'YearsCode', 'YearsCodePro', 'Employment', 'Hobbyist'])
#Removing rows without values
df_reg.dropna(inplace = True)
df_reg2.dropna(inplace = True)


#Verifying all values are numerical
column_values = df_reg[['YearsCodePro']].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)

column_values = df_reg[['YearsCode']].values
unique_values = np.unique(column_values)
print(unique_values)


#replacing string to int values
df_reg.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                  inplace=True)

df_reg2.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                  inplace=True)

#replacing Yes/No to numerical values
df_reg2.replace(to_replace={'No': '0',
                              'Yes': '1'},
                  inplace=True)

#setting appropriate data types
df_reg = df_reg.astype({'YearsCodePro': 'int64', 'YearsCode': 'int64'} , copy=False)
#df_reg = df_reg.astype('float64', copy=False)
print(df_reg['YearsCode'])
print(df_reg['YearsCodePro'])
print(df_reg2['Hobbyist'])

#one hot encoding for Employment column
Empl = LabelBinarizer().fit_transform(df_reg2.Employment)
print(Empl)

# df_reg2 = pd.concat([df_reg2, pd.get_dummies(df_reg2['Employment'])])
# print(df_reg2)

#Verifying correlation
print(df_reg.corr())


#plots
# plt.plot(df_reg['YearsCodePro'], df_reg['Age'],'ro', markersize=0.3)
# plt.ylabel('YearsCodePro')
# plt.xlabel('Age')
# plt.plot(df_reg['YearsCodePro'], df_reg['Hobbyist'],'ro', markersize=0.3, color = 'green')
# plt.ylabel('YearsCodePro')
# plt.xlabel('Hobbyist')
# plt.plot(df_reg['YearsCodePro'], Empl,'ro', markersize=0.3, color = 'yellow')
# plt.ylabel('YearsCodePro')
# plt.xlabel(Empl)
# plt.plot(df_reg['YearsCodePro'], df_reg['YearsCode'],'ro', markersize=0.3, color = 'blue')
# plt.ylabel('YearsCodePro')
# plt.xlabel('YearsCode/Age')
# plt.show()

print(df_reg.dtypes)

#Looking for outliers

sns.boxplot(y='YearsCodePro', data=df_reg)
plt.show();
sns.boxplot(y='YearsCode', data=df_reg)
plt.show();
sns.boxplot(y='Age', data=df_reg)
plt.show();

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_reg['YearsCode'], df_reg['YearsCodePro'])
ax.set_xlabel('YearsCode')
ax.set_ylabel('YearsCodePro')
plt.show()

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_reg['Age'], df_reg['YearsCodePro'])
ax.set_xlabel('Age')
ax.set_ylabel('YearsCodePro')
plt.show()

#z-score with absolute value
z = np.abs(stats.zscore(df_reg))
print(z)

#looking at outliers (>3)
threshold = 3
print(np.where(z > 3))

#keeping values within 3 standard deviations
df_reg_sd = df_reg[np.abs(df_reg - df_reg.mean()) <= 3*df_reg.std()]

sns.boxplot(y='YearsCodePro', data=df_reg_sd)
plt.show();
df_reg_sd.plot()
plt.show();

print(df_reg_sd.corr())

#interquartile range (IQR)/middle 50%

Q1 = df_reg.quantile(0.25)
Q3 = df_reg.quantile(0.75)
IQR = Q3 - Q1

df_reg_q = df_reg[~((df_reg < (Q1 - 1.5 * IQR)) | (df_reg > (Q3 + 1.5 * IQR))).any(axis=1)]

sns.boxplot(y='YearsCodePro', data=df_reg_q)
plt.show();
df_reg_q.plot()
plt.show();

print(df_reg_q.corr())

#linear regression

#one variable
reg = linear_model.LinearRegression()
reg.fit(df_reg[['YearsCode']], df_reg['YearsCodePro']);

print(reg.predict([[10]]))



#two variables
reg2 = linear_model.LinearRegression()
reg2.fit(df_reg[['YearsCode', 'Age']], df_reg['YearsCodePro']);

print(reg2.coef_)

print(reg2.predict(df_reg[['YearsCode', 'Age']]))

y_pred = reg2.predict(df_reg[['YearsCode', 'Age']])
y_true = df_reg['YearsCodePro']

print(y_true)
print(y_pred)
print(mean_squared_error(y_true, y_pred))


# reg3 = linear_model.LinearRegression()
# reg3.fit(df_reg2[['YearsCode', 'Age', 'Hobbyist', 'Employment']], df_reg2['YearsCodePro']);

# print(reg3.predict(df_reg2[['YearsCode', 'Age', 'Hobbyist', 'Employment']]))


