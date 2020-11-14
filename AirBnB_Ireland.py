
"""
Name:               Karina Jonina 
Github:             https://github.com/kjonina/
Data Gathered:      http://insideairbnb.com/get-the-data.html
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from io import StringIO # This is used for fast string concatination
import nltk # Use nltk for valid words
import collections as co # Need to make hash 'dictionaries' from nltk for fast processing
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer #Bag of Words

# read the CSV file
df = pd.read_csv('listings.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(df.head())

# checking the df shape
print(df.shape)
#(27131, 16)


# prints out names of columns
print(df.columns)
#Index(['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
#       'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
#       'minimum_nights', 'number_of_reviews', 'number_of_reviews_ltm',
#       'last_review', 'calculated_host_listings_count', 'availability_365'],
#      dtype='object')


# This tells us which variables are object, int64 and float 64. This would mean that 
# some of the object variables might have to be changed into a categorical variables and int64 to float64 
# depending on our analysis.
print(df.info())


# checking for missing data
df.isnull().sum() 
#id                                   0
#name                                 2
#host_id                              0
#host_name                            4
#neighbourhood_group                  0
#neighbourhood                        0
#latitude                             0
#longitude                            0
#room_type                            0
#price                                0
#minimum_nights                       0
#number_of_reviews                    0
#number_of_reviews_ltm                0
#last_review                       3802
#calculated_host_listings_count       0
#availability_365                     0

# getting the statistics such as the mean, standard deviation, min and max for numerical variables
print(df.describe()) 


df['neighbourhood'] = df['neighbourhood'].astype('category')
df['room_type'] = df['room_type'].astype('category')



# =============================================================================
# Droping Columns
# =============================================================================
# dropping Last Review 
df.drop('last_review',axis=1,inplace=True)

# =============================================================================
# Examining EDA
# =============================================================================
