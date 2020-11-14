
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
df['neighbourhood_group'] = df['neighbourhood_group'].astype('category')
df['room_type'] = df['room_type'].astype('category')



# =============================================================================
# Droping Columns
# =============================================================================
# dropping Last Review 
df.drop('last_review',axis=1,inplace=True)

# =============================================================================
# Examining EDA
# =============================================================================

df.groupby(['neighbourhood_group']).size().sort_values(ascending=False)
#Dublin City Council                      6102
#Kerry County Council                     2927
#Donegal County Council                   2044
#Cork County Council                      1881
#Galway County Council                    1656
#Clare County Council                     1532
#Mayo County Council                      1279
#Galway City Council                      1027
#Dun Laoghaire-rathdown County Council     829
#Fingal County Council                     762
#Wexford County Council                    721
#Wicklow County Council                    551
#Sligo County Council                      508
#Waterford City And County Council         505
#Cork City Council                         487
#Tipperary County Council                  456
#Kilkenny County Council                   435
#Meath County Council                      422
#Limerick City And County Council          421
#Louth County Council                      347
#Kildare County Council                    339
#South Dublin County Council               325
#Leitrim County Council                    270
#Westmeath County Council                  228
#Roscommon County Council                  204
#Cavan County Council                      193
#Carlow County Council                     175
#Laois County Council                      171
#Offaly County Council                     150
#Monaghan County Council                   128
#Longford County Council                    56

plt.figure(figsize = (12, 8))
sns.countplot(x = 'neighbourhood_group', data = df, palette = 'terrain',order = df['neighbourhood_group'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Number of AirBnBs in each County Councils', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Councils', fontsize = 14)
plt.show()



df.groupby(['neighbourhood']).size().sort_values(ascending=False)

df.groupby(['neighbourhood']).size().sort_values(ascending=False)
plt.figure(figsize = (12, 8))
sns.countplot(x = 'neighbourhood', data = df, palette = 'terrain',order = df['neighbourhood'].value_counts().head(50).index)
plt.xticks(rotation = 90)
plt.title('Number of AirBnBs in each Neighbourhood', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Councils', fontsize = 14)
plt.show()

# =============================================================================
# Splitting "neighbourhood_group" to just Dublin, Donegal and Cavan etc.
# =============================================================================

df['County1'] = (np.where(df['neighbourhood_group'].str.contains(' County'),
                  df['neighbourhood_group'].str.split(' County').str[0],
                  df['neighbourhood_group']))


df['County'] = (np.where(df['County1'].str.contains(' City'),
                  df['County1'].str.split(' City').str[0],
                  df['County1']))


df['Parish'] = (np.where(df['neighbourhood'].str.contains(' LEA'),
                  df['neighbourhood'].str.split(' LEA').str[0],
                  df['neighbourhood']))




plt.figure(figsize = (12, 8))
sns.countplot(x = 'County', data = df, palette = 'terrain',order = df['County'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Number of AirBnBs in each County', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('County', fontsize = 14)
plt.show()



df.groupby(['Parish']).size().sort_values(ascending=False)

df.groupby(['Parish']).size().sort_values(ascending=False)
plt.figure(figsize = (12, 8))
sns.countplot(x = 'Parish', data = df, palette = 'terrain',order = df['Parish'].value_counts().head(50).index)
plt.xticks(rotation = 90)
plt.title('Number of AirBnBs in each Top 50 Parish', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Parish', fontsize = 14)
plt.show()



'''
IDEAS FOR EDA

- boxplot for Councils
- map of the place by Longitude

- split data by lstrip to get Dublin, Cavan, Carlow,
- get a gradient map

- split data by LEA -
- get a gradient map again (especially of dublin!)


'''
# =============================================================================
# Checking Room Type
# =============================================================================
# checking for unique values
df.room_type.unique()
#array(['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'],
#      dtype=object)


# counting numbers of Room Types
df.groupby(['room_type']).size().sort_values(ascending=False)
#Entire home/apt    15433
#Private room       11139
#Hotel room           320
#Shared room          239

#Plot the graph of Room Type
plt.figure(figsize = (12, 8))
sns.countplot(x = 'room_type', data = df, palette = 'terrain',order = df['room_type'].value_counts().index)
plt.xticks(rotation = 90)
plt.title('Number of Room Types', fontsize = 16)
plt.ylabel('count', fontsize = 14)
plt.xlabel('Room Types', fontsize = 14)
plt.show()


# =============================================================================
# Checking Room Type by Availability 365 and Price
# =============================================================================

room_type_availablity = df.groupby(
        ['room_type']
        )['availability_365', 'price'].mean()

#                 availability_365       price
#room_type                                    
#Entire home/apt        154.516879  248.731873
#Hotel room             168.050000   96.550000
#Private room           130.476524   76.074064
#Shared room            115.962343   56.108787


#Plot the graph of Room Type by Availability 365 and Price
room_type_availablity.plot()

'''
CREATE A BAR CHART
'''


# =============================================================================
# Checking County by Availability 365 and Price
# =============================================================================

county_availablity_price = df.groupby(
        ['County']
        )['availability_365', 'price'].mean()

#                        availability_365       price
#County                                              
#Carlow                        155.222857  110.097143
#Cavan                         213.844560   93.373057
#Clare                         174.340078  126.587467
#Cork                          163.842061  111.277872
#Donegal                       201.514677  105.308219
#Dublin                         63.825139  323.215175
#Dun Laoghaire-rathdown         70.465621  118.004825
#Fingal                         98.812336  104.179790
#Galway                        162.500559  204.108461
#Kerry                         191.302016  144.601982
#Kildare                       141.466077  107.115044
#Kilkenny                      150.326437  133.818391
#Laois                         191.584795   85.637427
#Leitrim                       189.659259  115.388889
#Limerick                      174.066508  139.173397
#Longford                      159.464286  140.303571
#Louth                         198.273775  168.700288
#Mayo                          190.433151  107.325254
#Meath                         158.845972  160.869668
#Monaghan                      190.710938   90.984375
#Offaly                        177.660000  102.186667
#Roscommon                     192.745098  102.916667
#Sligo                         161.984252  101.708661
#South Dublin                   91.929231   80.667692
#Tipperary                     191.081140  125.421053
#Waterford                     149.324752  118.902970
#Westmeath                     195.881579  110.855263
#Wexford                       144.213592  140.382802
#Wicklow                       167.646098  108.903811

#plotting the data
plt.figure(figsize = (20, 16))
county_availablity_price.plot()
plt.title('Checking correlation between Availability and Price', fontsize = 16)
plt.ylabel('Availability', fontsize = 14)
plt.xlabel('Price', fontsize = 14)

#plotting the data
plt.figure(figsize = (12, 8))
sns.relplot(x = "price", y = "availability_365", data = room_type_availablity);
plt.title('Checking correlation between Availability and Price', fontsize = 16)
plt.ylabel('Availability', fontsize = 14)
plt.xlabel('Price', fontsize = 14)



sns.catplot(x="price", y="room_type", data = df)



'''
SCATTERPLOT FOR PRICE AND AVAILABILITY
'''

sns.relplot(x = "price", y = "availability_365", data = df, hue="County");

# =============================================================================
# 
# =============================================================================
#let's see what hosts (IDs) have the most listings on Airbnb platform and taking advantage of this service
top_host = df.host_id.value_counts().head(10)


#222188458    143
#218736839    113
#98759090     100
#105690532     81
#54449369      62
#148347815     58
#160402201     48
#208252367     44
#123745971     41
#134884575     38

top_host1 = df.groupby(
        ['host_id']
        )['availability_365', 'price'].mean()




#pivoting table to produce:
top_host_table_AV = pd.DataFrame({'A': df['host_id'],
                   'B': df['County'],
                   'C': df['availability_365']})

top_host_table_AV.A.value_counts().head(10)

'''TRYING TO CREATE A MEAN FOR AVAILABILITY FOR TOP 15 HOSTS'''

# =============================================================================
# Creating the map of Ireland
# =============================================================================

#let's what we can do with our given longtitude and latitude columns
#let's see how scatterplot will come out 
map_of_ROI=df.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
map_of_ROI.legend()
