
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
df['county'] = df['county'].str.replace('Dun Laoghaire-rathdown','Dublin')
df['county'] = df['county'].str.replace('Fingal','Dublin')

# dropping county1
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
# Examining EDA
# =============================================================================

sns.set()

#examining the data in Councils
print(df.groupby(['neighbourhood_group']).size().sort_values(ascending=False))


# create a graph
plt.figure(figsize = (12, 8))
sns.countplot(y = 'neighbourhood_group', data = df, palette = 'terrain',order = df['neighbourhood_group'].value_counts().index)
plt.title('Number of AirBnBs listings under each Councils\' Supervision', fontsize = 20)
plt.ylabel('Councils', fontsize = 14)
plt.xlabel('Number of AirBnBs', fontsize = 14)
plt.show()




# examining data in Parishes
print(df.groupby(['parish']).size().sort_values(ascending=False))

# create a graph for Parishes
plt.figure(figsize = (12, 8))
sns.countplot(y = 'parish', data = df, palette = 'terrain',order = df['parish'].value_counts().head(20).index)
plt.xticks(rotation = 90)
plt.title('Number of AirBnBs Listings in Top 20 Parish', fontsize = 20)
plt.ylabel('List of Parishes', fontsize = 14)
plt.xlabel('Number of AirBnBs', fontsize = 14)
plt.show()
plt.show()

# examining data in Parishes
print(df.groupby(['county']).size().sort_values(ascending=False))

#creating a graph of counties
plt.figure(figsize = (12, 8))
sns.countplot(y = 'county', data = df, palette = 'magma',order = df['county'].value_counts().index)
plt.title('Number of AirBnBs in Each County', fontsize = 20)
plt.ylabel('List of Country', fontsize = 14)
plt.xlabel('Number of AirBnBs', fontsize = 14)
plt.show()


# checking for unique values
df.room_type.unique()

# counting numbers of Room Types
df.groupby(['room_type']).size().sort_values(ascending=False)


#Plot the graph of Room Type
plt.figure(figsize = (12, 8))
sns.countplot(y = 'room_type', data = df, palette = 'terrain',order = df['room_type'].value_counts().index)
plt.title('Number of Room Types', fontsize = 20)
plt.ylabel('Room Type', fontsize = 14)
plt.xlabel('count', fontsize = 14)
plt.show()


# =============================================================================
# Checking correlations between numeric variables
# =============================================================================
variables = pd.DataFrame({
                   'price': df['price'],
                   'minimum_nights': df['minimum_nights'],
                   'number_of_reviews': df['number_of_reviews'],
                   'availability_365': df['availability_365']})

plt.figure(figsize = (12, 8))       
variables_matrix_heatmap = sns.heatmap(variables.corr(method = 'spearman'), annot = True, cmap = "viridis");



variables_matrix_heatmap.figure.savefig('variables_matrix_heatmap.png')

plt.figure(figsize = (12, 8)) 
plt.scatter(x= 'availability_365', y = 'price', data = df) 
plt.ylabel('availability_365', fontsize = 14)
plt.xlabel('price', fontsize = 14)
plt.show()


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

#examining the correlation again
plt.figure(figsize = (12, 8)) 
plt.scatter(x= 'availability_365', y = 'price', data = df) 
plt.ylabel('availability_365', fontsize = 14)
plt.xlabel('price', fontsize = 14)
plt.show()

#print("The dataset has {} rows and {} columns.".format(*df.shape))
#The dataset has 27130 rows and 18 columns.

# Decided that the data examined will be all under 1000
newdf = df.loc[df['price'].abs()<=1000]
#examining the correlation again
plt.figure(figsize = (12, 8)) 
plt.scatter(x= 'availability_365', y = 'price', data = newdf) 
plt.ylabel('availability_365', fontsize = 14)
plt.xlabel('price', fontsize = 14)
plt.show()

print("The dataset has {} rows and {} columns.".format(*newdf.shape))
#The dataset has 27027 rows and 18 columns.

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
plt.show()


# Average availability in each province
province_availability = province_price_avaibility.groupby('province')['availability_365'].mean()
print(province_availability)
#connacht    171.626933
#leinster     97.097119 - renting out for the summer?
#munster     175.928368
#ulster      201.878840

plt.figure(figsize = (12, 8))
room_availability_graph = sns.barplot(x = 'room_type', y= 'availability_365', data = room_availability.reset_index(), palette = 'terrain')

room_availability_graph.set_title('Availability for Each Room Type', fontsize = 20)
room_availability_graph.set_ylabel('Number of Room Types', fontsize = 14)
room_availability_graph.set_xlabel('Room Type', fontsize = 14)
plt.show()


#Save the graph
room_availability_graph.figure.savefig('room_availability_graph.png')



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
price_room_boxplot = sns.boxplot(x = 'room_type', y = 'price',
            data = room_price_avaibility, fliersize = 0)
price_room_boxplot.set_title('Price of Room per Night for Room Type', fontsize = 20)
price_room_boxplot.set_ylabel('Price [EUR]', fontsize = 14)
price_room_boxplot.set_xlabel('Room Type', fontsize = 14)
price_room_boxplot.set_ylim(0,300)
plt.show()

#Save the graph
price_room_boxplot.figure.savefig('price_room_boxplot.png')



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

room_availability_graph.set_title('Availability for Each Room Type', fontsize = 20)
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
                   'county': newdf['county'],
                   'availability_365': newdf['availability_365'],
                   'price': newdf['price']})

#examining the number of listings in each parish
plt.figure(figsize = (12, 12))
parish_listings = sns.countplot(y = 'parish', data = parish_price_availability, palette = 'terrain',order = parish_price_availability['parish'].value_counts().head(50).index)
parish_listings.set_title('Number of AirBnBs in each Parish', fontsize = 20)
parish_listings.set_ylabel('Name of Parish', fontsize = 12)
parish_listings.set_xlabel('Number of AirBnB listings',  fontsize = 12)
plt.show()

#Save the graph
parish_listings.figure.savefig('parish_listings.png')



#examining which parishes have the HIGHEST AVERAGE PRICE per listing (per night)
parish_price_availability.groupby('parish')['price'].mean().sort_values(ascending=False).head(10)
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
price_parish_boxplot = sns.boxplot(x = 'parish', y = 'price',
            data = parish_price_availability, fliersize = 0,order = parish_price_availability['parish'].value_counts().head(10).index)
price_parish_boxplot.set_title('Booking Price per Night for Top 10 Parishes', fontsize = 20)
price_parish_boxplot.set_ylabel('Price [EUR]', fontsize = 14)
price_parish_boxplot.set_xlabel('Parish', fontsize = 14)
price_parish_boxplot.set_ylim(0,400)
price_parish_boxplot.set_xticklabels(price_parish_boxplot.get_xticklabels(), rotation = 45)
plt.show()

#Save the graph
price_parish_boxplot.figure.savefig('price_parish_boxplot.png')


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
# Checking county by Availability 365 and Price
# =============================================================================

county_availablity_price = newdf.groupby(
        ['county']
        )['availability_365', 'price']


print(county_availablity_price.describe())


#plotting the data
plt.figure(figsize = (20, 16))
county_availablity_price.plot()
plt.title('Checking correlation between Availability and Price', fontsize = 20)
plt.ylabel('Availability', fontsize = 14)
plt.xlabel('Price', fontsize = 14)



#plotting the data
plt.figure(figsize = (12, 8))
sns.relplot(x = "price", y = "availability_365", data = county_availablity_price);
plt.title('Checking correlation between Availability and Price', fontsize = 20)
plt.ylabel('Availability', fontsize = 14)
plt.xlabel('Price', fontsize = 14)


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


sns.catplot(x="price", y="availability_365", data = newdf_dublin)


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