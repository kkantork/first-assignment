#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 04:05:00 2020

@author: rotnak
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

survey_df = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent', 'WorkWeekHrs', 'CodeRevHrs'],
                        index_col='Respondent')

survey_df.dropna(inplace=True)

survey_df.dtypes

survey_df = survey_df.astype('int64', copy=False)

survey_df = survey_df[survey_df['WorkWeekHrs'] < 100]

plt.plot(survey_df['WorkWeekHrs'], survey_df['CodeRevHrs'], 'ro', markersize = 0.3)
plt.ylabel('CodeRevHrs')
plt.xlabel('WorkWeekHrs')
plt.show()

survey_df2 = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent', 'WorkWeekHrs', 'CodeRevHrs', 'Hobbyist'],
                        index_col='Respondent')

hobb = survey_df2[survey_df2['Hobbyist'] == 'Yes']

hobb.dropna(inplace=True)

hobb = hobb.astype({'WorkWeekHrs': 'int64', 'CodeRevHrs': 'int64'} , copy=False)

hobb = hobb[hobb['WorkWeekHrs'] < 100]

plt.plot(hobb['WorkWeekHrs'], hobb['CodeRevHrs'], 'ro', markersize = 0.3)
plt.ylabel('CodeRevHrs')
plt.xlabel('WorkWeekHrs')
plt.show()

nonhobb = survey_df2[survey_df2['Hobbyist'] == 'No']

nonhobb.dropna(inplace=True)

nonhobb = nonhobb.astype({'WorkWeekHrs': 'int64', 'CodeRevHrs': 'int64'} , copy=False)

print(nonhobb.dtypes)

nonhobb = nonhobb[nonhobb['WorkWeekHrs'] < 100]

plt.plot(nonhobb['WorkWeekHrs'], nonhobb['CodeRevHrs'], 'ro', markersize = 0.3)
plt.ylabel('CodeRevHrs')
plt.xlabel('WorkWeekHrs')
plt.show()