
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
from scipy import stats

# read the CSV file
df = pd.read_csv('listings.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(df.head())

# checking shape
print("The dataset has {} rows and {} columns.".format(*df.shape))

# ... and duplicates
print("It contains {} duplicates.".format(df.duplicated().sum()))

# prints out names of columns
print(df.columns)



# This tells us which variables are object, int64 and float 64. 
print(df.info())

# checking for missing data
df.isnull().sum() 

# getting the statistics such as the mean, standard deviation, min and max for numerical variables
print(df.describe()) 


# =============================================================================
# Changing variables dtypes
# =============================================================================
df['neighbourhood'] = df['neighbourhood'].astype('category')
df['neighbourhood_group'] = df['neighbourhood_group'].astype('category')
df['room_type'] = df['room_type'].astype('category')



# =============================================================================
# Droping Columns
# =============================================================================
# dropping Last Review 
df.drop('last_review',axis=1,inplace=True)

# =============================================================================
# Splitting "neighbourhood_group" to just Co. Dublin, Donegal and Cavan etc.
# =============================================================================
df.neighbourhood_group.unique()

df['county1'] = (np.where(df['neighbourhood_group'].str.contains(' County'),
                  df['neighbourhood_group'].str.split(' County').str[0],
                  df['neighbourhood_group']))


df['county2'] = (np.where(df['county1'].str.contains(' City'),
                  df['county1'].str.split(' City').str[0],
                  df['county1']))

#Removing 'South' in Dublin
df['county'] = (np.where(df['county2'].str.contains('South '),
                  df['county2'].str.split('South ').str[1],
                  df['county2']))

#Must change Dun Laoighre and Fingal to Dublin for it to be County
df.neighbourhood_group.unique()

# dropping county1
df['county'] = df['county'].str.replace('Dun Laoghaire-rathdown','Dublin')
df['county'] = df['county'].str.replace('Fingal','Dublin')
df.drop('county1',axis=1,inplace=True)

# dropping county2
df.drop('county2',axis=1,inplace=True)

''' I am sure there is a lot more smooth method to create county'''

# =============================================================================
# Removing 'LEA' from Parishes
# =============================================================================
#Creating Parishes 
df['parish'] = (np.where(df['neighbourhood'].str.contains(' LEA'),
                  df['neighbourhood'].str.split(' LEA').str[0],
                  df['neighbourhood']))


# =============================================================================
# Checking out Outlier in Price
# =============================================================================
sns.set()


plt.figure(figsize = (12, 8))
spotting_outliers = plt.scatter(x= 'availability_365', y = 'price', data = df) 
plt.title('Checking for outliers in price',  fontsize = 20)
plt.ylabel('Price [EUR]', fontsize = 14)
plt.xlabel('availability_365', fontsize = 14)
plt.show()

#Save the graph
spotting_outliers.figure.savefig('spotting_outliers.png')



# =============================================================================
# Eliminating Outlier in Price
# =============================================================================
print("The dataset has {} rows and {} columns.".format(*df.shape))
#The dataset has 27131 rows and 17 columns.

#Normalising the price 
df['z_score']=stats.zscore(df['price'])

#checking the outlier
outlier = df.loc[df['z_score'].abs()>3]
print(outlier)
# The outlier shown is correct. THe price is 1173721

#Removing that single outlier 
df = df.loc[df['z_score'].abs()<=3]

print("The dataset has {} rows and {} columns.".format(*df.shape))
#The dataset has 27130 rows and 18 columns.



#examining the correlation again
plt.figure(figsize = (12, 8)) 
spotting_outliers2 = plt.scatter(x= 'availability_365', y = 'price', data = df) 
plt.title('Examining other outliers in the dataset in Price', fontsize = 20)
plt.ylabel('Price [EUR]', fontsize = 14)
plt.xlabel('availability_365', fontsize = 14)
plt.show()

spotting_outliers2.figure.savefig('spotting_outliers2.png')



#Normalising the price 
df['z_score']=stats.zscore(df['price'])

#checking the outlier
outlier = df.loc[df['z_score'].abs()>3]
print(outlier)
# The outlier shown is correct. THe price is 1173721

#Removing that single outlier 
newdf = df.loc[df['z_score'].abs()<=3]

#examining the correlation again
plt.figure(figsize = (12, 8)) 
final_data = plt.scatter(x= 'availability_365', y = 'price', data = newdf) 
plt.title('Final Data After Removing all Z-Scores above 3', fontsize = 20)
plt.ylabel('availability_365', fontsize = 14)
plt.xlabel('Price [EUR]', fontsize = 14)
plt.show()

final_data.figure.savefig('final_data.png')


print("The dataset has {} rows and {} columns.".format(*newdf.shape))
#The dataset has 27082 rows and 18 columns.

# =============================================================================
# Examining EDA
# =============================================================================

#examining the data in Councils
print(newdf.groupby(['neighbourhood_group']).size().sort_values(ascending=False))


# create a graph
plt.figure(figsize = (12, 8))
neighbourhood_group_graph = sns.countplot(y = 'neighbourhood_group', data = newdf, palette = 'terrain',order = newdf['neighbourhood_group'].value_counts().index)
neighbourhood_group_graph.set_title('Number of AirBnBs listings under each Councils\' Supervision', fontsize = 20)
neighbourhood_group_graph.set_ylabel('Council', fontsize = 14)
neighbourhood_group_graph.set_xlabel('Number of AirBnB Listings', fontsize = 14)
plt.show()

#Save the graph
neighbourhood_group_graph.figure.savefig('neighbourhood_group_graph.png')


# examining data in Parishes
print(newdf.groupby(['parish']).size().sort_values(ascending=False))

# create a graph for Parishes
plt.figure(figsize = (12, 8))
parish_graph = sns.countplot(y = 'parish', data = newdf, palette = 'terrain',order = newdf['parish'].value_counts().head(20).index)
parish_graph.set_title('Number of AirBnBs Listings in Top 20 Parish', fontsize = 20)
parish_graph.set_ylabel('List of Parishes', fontsize = 14)
parish_graph.set_xlabel('Number of AirBnB Listings', fontsize = 14)
plt.show()

#Save the graph
parish_graph.figure.savefig('parish_graph.png')


# examining data in Parishes
print(newdf.groupby(['county']).size().sort_values(ascending=False))

#creating a graph of counties
plt.figure(figsize = (12, 8))
county_graph = sns.countplot(y = 'county', data = newdf, palette = 'magma',order = newdf['county'].value_counts().index)
county_graph.set_title('Number of AirBnBs Listings in Each County', fontsize = 20)
county_graph.set_ylabel('List of Counties', fontsize = 14)
county_graph.set_xlabel('Number of AirBnB Listings', fontsize = 14)
plt.show()

#Save the graph
parish_graph.figure.savefig('parish_graph.png')



# checking for unique values
newdf.room_type.unique()

# counting numbers of Room Types
newdf.groupby(['room_type']).size().sort_values(ascending=False)


#Plot the graph of Room Type
plt.figure(figsize = (12, 8))
room_graph = sns.countplot(x= 'room_type', data = newdf, palette = 'terrain',order = newdf['room_type'].value_counts().index)
room_graph.set_title('Number of AirBnBs Listings for Each Room Type', fontsize = 20)
room_graph.set_xlabel('Room Type', fontsize = 14)
room_graph.set_ylabel('Number of AirBnB Listings', fontsize = 14)
plt.show()

#Save the graph
room_graph.figure.savefig('room_graph.png')


# =============================================================================
# Checking correlations between numeric variables
# =============================================================================
variables = pd.DataFrame({
                   'price': newdf['price'],
                   'minimum_nights': newdf['minimum_nights'],
                   'number_of_reviews': newdf['number_of_reviews'],
                   'availability_365': newdf['availability_365']})

plt.figure(figsize = (12, 8))       
variables_matrix_heatmap = sns.heatmap(variables.corr(method = 'spearman'), annot = True, cmap = "viridis");



variables_matrix_heatmap.figure.savefig('variables_matrix_heatmap.png')

plt.figure(figsize = (12, 8)) 
plt.scatter(x= 'availability_365', y = 'price', data = newdf) 
plt.ylabel('availability_365', fontsize = 14)
plt.xlabel('price', fontsize = 14)
plt.show()

# =============================================================================
# Creating new variable: province 
# =============================================================================
# trying to create the Connacht
newdf.loc[newdf['county'].str.contains('Mayo'), 'province'] = 'Connacht'
newdf.loc[newdf['county'].str.contains('Sligo'), 'province'] = 'Connacht'
newdf.loc[newdf['county'].str.contains('Galway'), 'province'] = 'Connacht'
newdf.loc[newdf['county'].str.contains('Roscommon'), 'province'] = 'Connacht'
newdf.loc[newdf['county'].str.contains('Leitrim'), 'province'] = 'Connacht'


newdf.loc[newdf['county'].str.contains('Cork'), 'province'] = 'Munster'
newdf.loc[newdf['county'].str.contains('Kerry'), 'province'] = 'Munster'
newdf.loc[newdf['county'].str.contains('Waterford'), 'province'] = 'Munster'
newdf.loc[newdf['county'].str.contains('Clare'), 'province'] = 'Munster'
newdf.loc[newdf['county'].str.contains('Limerick'), 'province'] = 'Munster'
newdf.loc[newdf['county'].str.contains('Tipperary'), 'province'] = 'Munster'

newdf.loc[newdf['county'].str.contains('Dublin'), 'province'] = 'Leinster'#1
newdf.loc[newdf['county'].str.contains('Louth'), 'province'] = 'Leinster' #2
newdf.loc[newdf['county'].str.contains('Offaly'), 'province'] = 'Leinster'#3
newdf.loc[newdf['county'].str.contains('Wicklow'), 'province'] = 'Leinster'#4
newdf.loc[newdf['county'].str.contains('Wexford'), 'province'] = 'Leinster'#5
newdf.loc[newdf['county'].str.contains('Laois'), 'province'] = 'Leinster'#6
newdf.loc[newdf['county'].str.contains('Kildare'), 'province'] = 'Leinster'#7
newdf.loc[newdf['county'].str.contains('Kilkenny'), 'province'] = 'Leinster'#8
newdf.loc[newdf['county'].str.contains('Longford'), 'province'] = 'Leinster'#9
newdf.loc[newdf['county'].str.contains('Meath'), 'province'] = 'Leinster'#10
newdf.loc[newdf['county'].str.contains('Westmeath'), 'province'] = 'Leinster'#11
newdf.loc[newdf['county'].str.contains('Carlow'), 'province'] = 'Leinster'#12

newdf.loc[newdf['county'].str.contains('Cavan'), 'province'] = 'Ulster'#1
newdf.loc[newdf['county'].str.contains('Donegal'), 'province'] = 'Ulster'#2
newdf.loc[newdf['county'].str.contains('Monaghan'), 'province'] = 'Ulster'#3


newdf['province'].value_counts().sort_values(ascending=False)
#leinster    11316
#munster      8069
#connacht     4849
#ulster       2344


plt.figure(figsize = (12, 8))
province_graph = sns.countplot(x = 'province', data = newdf, palette = 'terrain',order = newdf['province'].value_counts().index)
province_graph.set_title('Number of AirBnB Listings in Each Province', fontsize = 20)
province_graph.set_ylabel('Number of of AirBnB Listings', fontsize = 14)
province_graph.set_xlabel('Province', fontsize = 14)
plt.show()

#Save the graph
province_graph.figure.savefig('province_graph.png')

# =============================================================================
# Checking price and availablitiy for each province
# =============================================================================
province_price_avaibility = pd.DataFrame({'province': newdf['province'],
                   'availability_365': newdf['availability_365'],
                   'price': newdf['price']})



province_price_avaibility.groupby('province')['price'].describe()

# boxplot for  price for the room 
plt.figure(figsize = (12, 8))
price_province_boxplot = sns.boxplot(x = 'province', y = 'price',
            data = province_price_avaibility, fliersize = 0)
price_province_boxplot.set_title('Price of Room per Night for Room Type in each Province', fontsize = 20)
price_province_boxplot.set_ylabel('Price [EUR]', fontsize = 14)
price_province_boxplot.set_xlabel('Province', fontsize = 14)
price_province_boxplot.set_ylim(0,300)
plt.show()


# Average availability in each province
province_availability = province_price_avaibility.groupby('province')['availability_365'].mean()
print(province_availability)
#connacht    171.626933
#leinster     97.097119 - renting out for the summer?
#munster     175.928368
#ulster      201.878840

plt.figure(figsize = (12, 8))
province_availability_graph = sns.barplot(x = 'province', y= 'availability_365', data = province_availability.reset_index(), palette = 'terrain')
province_availability_graph.set_title('Average Availability in Each Province', fontsize = 20)
province_availability_graph.set_ylabel('Number of Room Types', fontsize = 14)
province_availability_graph.set_xlabel('Room Type', fontsize = 14)
plt.show()


#Save the graph
province_availability_graph.figure.savefig('province_availability_graph.png')



# =============================================================================
# Checking price and availablitiy for each room type
# =============================================================================
# creating a new dataset with Room Type and 
room_price_avaibility = pd.DataFrame({'room_type': newdf['room_type'],
                   'availability_365': newdf['availability_365'],
                   'price': newdf['price']})


room_price_avaibility.groupby('room_type')['price'].describe()
#                   count        mean         std   min   25%    50%     75%        max
#room_type                                                                     
#Entire home/apt  15349.0  138.737703  104.924566  11.0  80.0  110.0  155.00    1000.0  
#Hotel room         320.0   96.550000   67.870132   0.0  62.0   82.0  107.25     500.0    
#Private room     11119.0   66.937494   57.303045  12.0  40.0   55.0   75.00    1000.0    
#Shared room        239.0   56.108787  109.101972  10.0  20.0   30.0   52.50     999.0    

 

# boxplot for  price for the room 
plt.figure(figsize = (12, 8))
room_price_boxplot = sns.boxplot(x = 'room_type', y = 'price',
            data = room_price_avaibility, fliersize = 0)
room_price_boxplot.set_title('Price of Room per Night for Room Type', fontsize = 20)
room_price_boxplot.set_ylabel('Price [EUR]', fontsize = 14)
room_price_boxplot.set_xlabel('Room Type', fontsize = 14)
room_price_boxplot.set_ylim(0,300)
plt.show()

#Save the graph
room_price_boxplot.figure.savefig('room_price_boxplot.png')



# Average availability for each room type
room_availability = room_price_avaibility.groupby('room_type')['availability_365'].mean()
print(room_availability)
#Entire home/apt    154.516879
#Hotel room         168.050000
#Private room       130.476524
#Shared room        115.962343
#Name: availability_365, dtype: float64


plt.figure(figsize = (12, 8))
room_availability_graph = sns.barplot(x = 'room_type', y= 'availability_365', data = room_availability.reset_index(), palette = 'terrain')
room_availability_graph.set_title('Average Availability for Each Room Type', fontsize = 20)
room_availability_graph.set_ylabel('Number of Room Types', fontsize = 14)
room_availability_graph.set_xlabel('Room Type', fontsize = 14)
plt.show()


#Save the graph
room_availability_graph.figure.savefig('room_availability_graph.png')

# =============================================================================
# Checking price and availablitiy for each parish
# =============================================================================
# checking AirBnBs per Parishes
parish_price_availability = pd.DataFrame({'parish': newdf['parish'],
                   'availability_365': newdf['availability_365'],
                   'price': newdf['price']})



parish_price_availability['parish'].unique()


#examining the number of listings in each parish
plt.figure(figsize = (12, 12))
parish_listings = sns.countplot(y = 'parish', data = parish_price_availability, palette = 'terrain',order = parish_price_availability['parish'].value_counts().head(50).index)
parish_listings.set_title('Number of AirBnBs in each Parish', fontsize = 20)
parish_listings.set_ylabel('Name of Parish', fontsize = 12)
parish_listings.set_xlabel('Number of AirBnB listings',  fontsize = 12)
plt.show()

#Save the graph
parish_listings.figure.savefig('parish_listings.png')

parish_price_availability['parish'].unique()



#examining which parishes have the HIGHEST AVERAGE PRICE per listing (per night)
top10_parish_average_price = parish_price_availability.groupby('parish')['price'].mean().sort_values(ascending=False).head(10)
#Ongar                    170.854545
#Dundalk-Carlingford      164.774566
#Bandon - Kinsale         151.003289
#Pembroke                 147.667697
#Carrick-on-Shannon       139.821138
#Cobh                     139.042735
#South East Inner City    134.120267
#Callan-Thomastown        133.887574
#Galway City Central      132.659170
#Moate                    130.716418

#Most parishes with the most number of listings
parish_price_availability['parish'].value_counts().head(10)
#North Inner City         1528
#South East Inner City    1347
#Kenmare                  1097
#South West Inner City     821
#Ennistimon                816
#Conamara North            746
#Corca Dhuibhne            739
#Pembroke                  647
#Killarney                 637
#Galway City Central       578

# boxplot for  price for the hotel type for Per Person
plt.figure(figsize = (12, 8))
parish_price_boxplot = sns.boxplot(x = 'parish', y = 'price',
            data = parish_price_availability, fliersize = 0,order = parish_price_availability['parish'].value_counts().head(10).index)
parish_price_boxplot.set_title('Booking Price per Night for Top 10 Parishes', fontsize = 20)
parish_price_boxplot.set_ylabel('Price [EUR]', fontsize = 14)
parish_price_boxplot.set_xlabel('Parish', fontsize = 14)
parish_price_boxplot.set_ylim(0,400)
parish_price_boxplot.set_xticklabels(parish_price_boxplot.get_xticklabels(), rotation = 45)
plt.show()

#Save the graph
parish_price_boxplot.figure.savefig('parish_price_boxplot.png')


''' NEEDS FIXING!
# Average availability for the TOP 10 parishes
parish_availability = parish_price_availability.groupby('parish')['availability_365'].mean().sort_values(ascending=False).head(10)
print(parish_availability)

plt.figure(figsize = (200, 200))
parish_availability_graph = sns.barplot(x = 'parish', y= 'availability_365', data = parish_price_availability)
parish_availability_graph.set_title('Availability for Top 10 Parishes', fontsize = 20)
parish_availability_graph.set_ylabel('Number of Available Nights', fontsize = 14)
parish_availability_graph.set_xlabel('Parish', fontsize = 14)
parish_availability_graph.set_xticklabels(parish_availability_graph.get_xticklabels(), rotation = 45)
parish_availability_graph.set_ylim(0,300)
plt.show()


#Save the graph
parish_availability_graph.figure.savefig('parish_availability_graph.png')
'''

# =============================================================================
# Price and avaialblity for each country
# =============================================================================

# checking AirBnBs per Parishes
county_price_availability = pd.DataFrame({'county': newdf['county'],
                   'availability_365': newdf['availability_365'],
                   'price': newdf['price']})

county_price_availability.groupby('county')['price'].describe()
#            count        mean         std   min   25%   50%     75%     max
#county                                                                     
#Carlow      175.0  110.097143  167.123971  20.0  45.0  76.0  105.50  1500.0
#Cavan       193.0   93.373057   94.213221  25.0  50.0  70.0  103.00  1000.0
#Clare      1531.0  113.157413   97.639791  11.0  65.0  89.0  130.00  1714.0
#Cork       2366.0  108.564243  102.714576  17.0  55.0  81.0  123.75  1175.0
#Donegal    2043.0  104.355849   83.485084  19.0  62.0  88.0  121.00  1500.0
#Dublin     8003.0  112.086717  111.636336   0.0  50.0  80.0  129.50  1757.0
#Galway     2671.0  111.821415  116.667426  12.0  56.0  85.0  124.00  1500.0
#Kerry      2923.0  116.277797   86.679523  17.0  70.0  96.0  137.00  1243.0
#Kildare     338.0   92.639053  125.752170   0.0  45.0  66.0  100.00  1500.0
#Kilkenny    435.0  133.818391  180.632176  15.0  60.0  83.0  125.00  1450.0
#Laois       171.0   85.637427   98.896664  24.0  40.0  60.0  100.00  1000.0
#Leitrim     270.0  115.388889   95.413017  20.0  60.0  80.5  135.75   614.0
#Limerick    419.0   84.942721  105.309144  15.0  44.0  65.0   99.00  1350.0
#Longford     55.0  106.490909  189.248501  20.0  44.0  70.0   94.00  1095.0
#Louth       345.0  140.597101  150.291374  20.0  63.0  98.0  154.00  1625.0
#Mayo       1278.0  105.061815   98.243342  19.0  58.0  85.0  120.00  1395.0
#Meath       420.0  105.900000  148.547300  19.0  45.0  71.0  111.00  1500.0
#Monaghan    128.0   90.984375   65.858472  21.0  45.0  67.0  122.25   320.0
#Offaly      149.0   89.449664   70.248924  12.0  50.0  71.0  100.00   540.0
#Roscommon   204.0  102.916667   88.096219  13.0  50.0  75.0  125.25   604.0
#Sligo       507.0   97.767258   86.109712  15.0  60.0  80.0  115.50  1440.0
#Tipperary   456.0  125.421053  189.944602  12.0  50.0  75.0  120.00  1729.0
#Waterford   504.0  108.039683   80.444728  18.0  60.0  90.0  130.00   660.0
#Westmeath   228.0  110.855263  120.400181  22.0  55.0  74.0  130.00  1143.0
#Wexford     719.0  106.824757   84.556041  25.0  60.0  89.0  125.00  1143.0
#Wicklow     551.0  108.903811  130.203211   0.0  54.0  79.0  120.00  1816.0

 

# boxplot for  price for the room 
plt.figure(figsize = (80, 8))
county_price_boxplot = sns.boxplot(x = 'county', y = 'price',
            data = county_price_availability, fliersize = 0)
county_price_boxplot.set_title('Price of Room per Night in Each County', fontsize = 20)
county_price_boxplot.set_ylabel('Price [EUR]', fontsize = 14)
county_price_boxplot.set_xlabel('Country', fontsize = 14)
county_price_boxplot.set_ylim(0,300)
plt.show()

#Save the graph
county_price_boxplot.figure.savefig('county_price_boxplot.png')



# Average availability for each room type
county_availability = county_price_availability.groupby('county')['availability_365'].mean().sort_values(ascending=False)
print(county_availability)
#Cavan        213.844560
#Donegal      201.613314
#Louth        199.423188
#Westmeath    195.881579
#Roscommon    192.745098
#Laois        191.584795
#Kerry        191.127609
#Tipperary    191.081140
#Monaghan     190.710938
#Mayo         190.501565
#Leitrim      189.659259
#Offaly       176.577181
#Clare        174.453298
#Limerick     173.205251
#Wicklow      167.646098
#Cork         163.675402
#Galway       162.411831
#Sligo        161.842209
#Meath        158.119048
#Longford     157.454545
#Carlow       155.222857
#Kilkenny     150.326437
#Waterford    149.537698
#Wexford      143.876217
#Kildare      141.106509
#Dublin        68.934650
#Name: availability_365, dtype: float64




plt.figure(figsize = (12, 8))
county_availability_graph = sns.barplot(x= "county", y = "availability_365", data = county_availability.reset_index(),
                 palette = 'magma')
county_availability_graph.set_title('Average Availability in Each County', fontsize = 20)
county_availability_graph.set_ylabel('Average Availability in Each County', fontsize = 14)
county_availability_graph.set_xlabel('Counties', fontsize = 14)
county_availability_graph.set_xticklabels(county_availability_graph.get_xticklabels(), rotation=45)
plt.show()


#Save the graph
county_availability_graph.figure.savefig('county_availability_graph.png')


'''
1 CREATE A HEATMAP OF NUMBER OF AIRBNB LISTINGS IN EACH COUNTY
2 STEM WORDS
3 RUN A LOGISTICAL REGRESSION FOR PRICE
'''


# =============================================================================
# Perhaps lack of correlation due to Dublin and the other counties
# =============================================================================
# Getting Dublin
newdf_dublin = newdf[newdf.county == 'Dublin']


newdf_dublin['neighbourhood_group'].unique()

newdf_dublin.parish.unique()
#array(['Ballyfermot-Drimnagh', 'South East Inner City',
#       'North Inner City', 'South West Inner City', 'Cabra-Glasnevin',
#       'Pembroke', 'Kimmage-Rathmines', 'Clontarf', 'Artane-Whitehall',
#       'Donaghmede', 'Ballymun-Finglas', 'Tallaght Central',
#       'Palmerstown-Fonthill', 'Clondalkin', 'Lucan',
#       'Rathfarnham-Templeogue', 'Tallaght South',
#       'Firhouse-Bohernabreena', 'Howth-Malahide',
#       'Blanchardstown-Mulhuddart', 'Balbriggan', 'Swords', 'Ongar',
#       'Castleknock', 'Dundrum', 'Stillorgan', 'Killiney-Shankill',
#       'Glencullen-Sandyford', 'Blackrock', 'Dún Laoghaire', 'Rush-Lusk'],
#      dtype=object)


# counting numbers of Room Types
newdf_dublin.groupby(['parish']).size().sort_values(ascending=False)
#North Inner City             1531
#South East Inner City        1350
#South West Inner City         828
#Pembroke                      648
#Cabra-Glasnevin               515
#Kimmage-Rathmines             408
#Clontarf                      306
#Howth-Malahide                226
#Dún Laoghaire                 211
#Artane-Whitehall              197
#Swords                        156
#Glencullen-Sandyford          142
#Ballyfermot-Drimnagh          133
#Blackrock                     130
#Dundrum                       124
#Stillorgan                    120
#Castleknock                   115
#Donaghmede                    104
#Killiney-Shankill             102
#Rathfarnham-Templeogue         89
#Ballymun-Finglas               82
#Blanchardstown-Mulhuddart      79
#Rush-Lusk                      70
#Clondalkin                     62
#Balbriggan                     61
#Ongar                          55
#Firhouse-Bohernabreena         50
#Lucan                          34
#Palmerstown-Fonthill           33
#Tallaght South                 33
#Tallaght Central               24


#Checking Dublin's Statistics
newdf_dublin.describe()


#plotting the data
plt.figure(figsize = (12, 8))
sns.relplot(x = "price", y = "availability_365", data = newdf_dublin);
plt.title('Checking correlation between Availability and Price', fontsize = 20)
plt.ylabel('Availability', fontsize = 14)
plt.xlabel('Price', fontsize = 14)

# =============================================================================
# Examining Top Hosts in ROI
# =============================================================================
#let's see what hosts (IDs) have the most listings on Airbnb platform and taking advantage of this service
top_host = newdf.host_id.value_counts().head(10)


#Creating a dataframe for top_host
top_host_newdf=pd.DataFrame(top_host)
top_host_newdf.reset_index(inplace=True)
top_host_newdf.rename(columns={'index':'Host_ID', 'host_id':'P_Count'}, inplace=True)
top_host_newdf


#Creating a graph for Hosts
host_graph=sns.barplot(x="Host_ID", y="P_Count", data=top_host_newdf,
                 palette='Blues_d')
host_graph.set_title('Hosts with the most listings in ROI')
host_graph.set_ylabel('Count of listings')
host_graph.set_xlabel('Host IDs')
host_graph.set_xticklabels(host_graph.get_xticklabels(), rotation=45)




'''TRYING TO CREATE A MEAN FOR AVAILABILITY FOR TOP 15 HOSTS'''
top_host1 = newdf.groupby(
        ['host_id']
        )['availability_365', 'price'].mean()




#pivoting table to produce:
top_host_table_AV = pd.DataFrame({'A': newdf['host_id'],
                   'B': newdf['county'],
                   'C': newdf['availability_365']})

top_host_table_AV.A.value_counts().head(10)


# =============================================================================
# Creating the map of Ireland
# =============================================================================

#let's what we can do with our given longtitude and latitude columns
#let's see how scatterplot will come out 
map_of_ROI=df.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
map_of_ROI.legend()


'''
IDEAS FOR EDA

- boxplot for Councils

- get a gradient map by county

- get a gradient map again (especially of dublin!)


'''