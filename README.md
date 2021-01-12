# AirBnB_Ireland
Conducting simple EDA and trying to predict AirBnB price 

# AirBnB Background
Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present a more unique, personalized way of experiencing the world. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Data analysis on millions of listings provided through Airbnb is a crucial factor for the company. These millions of listings generate a lot of data - data that can be analyzed and used for security, business decisions, understanding of customers' and providers' (hosts) behavior and performance on the platform, guiding marketing initiatives, implementation of innovative additional services and much more.


## Data Source

This dataset was collected from [AirBnb website](http://insideairbnb.com/get-the-data.html). It examines AirBnB listings for all of Ireland. This dataset has around 27131 observations in it with 16 columns and it is a mix between categorical and numeric values. 


## Business Questions:
The following Business Questions were thought of:

- [ ] What County and City Coucils have the most AirBnB listing?
- [ ] What County and City Coucils have the highest average price?
- [ ] What County and City Coucils have the highest average availability?

- [ ] What parishes have the most AirBnB listing?
- [ ] What parishes have the highest average price?
- [ ] What parishes have the highest average availability?

- [ ] What counties have the most AirBnB listing?
- [ ] What counties have the highest average price?
- [ ] What counties have the highest average availability?

- [ ] What provinces have the most AirBnB listing?
- [ ] What provinces have the highest average price?
- [ ] What provinces have the highest average availability?

- [ ] What is the most common room type listed in Ireland?
- [ ] What is the average price for listings in each room type?

- [ ] What are the most common words in the listings?

- [ ] Create a model to predict price of an AirBnB listing.




# Learning Outcomes

The purpose of this analysis for myself is to: 
- [ ] deal with parts of code I struggle with: splitting text data, creating loops.
- [ ] deal with outliers in prices
- [ ] create a more fancy looking graphs 
- [ ] run analysis on text column called ['Name'] 
- [ ] create a WordCloud for ['Name']
- [ ] create a predictive model for Prices 


## Data preparation
The following are variable 
['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
'minimum_nights', 'number_of_reviews', 'number_of_reviews_ltm',
'last_review', 'calculated_host_listings_count', 'availability_365']

The dataset has 27131 rows and 17 columns.

## Outliers
![spotting_outliers](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/spotting_outliers.png)
It is clear from the scatterplot that there is a massive outlier. The price for this entire hourse/ apartment in Kimmage-Rathmines (Co. Dublin)is €1173721. The house/ apartment is available for 180 nights per year and the minimum stay is 3 nights. It is unclear whether this is a typing error on behalf of the owner.

![spotting_outliers2](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/spotting_outliers2.png)
When the outlier is removed, here are the rest of the prices for the rest of the dateset.

![final_data](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/final_data.png)
After running z-scores and removing all outliers above 3, there is 27082 rows and 18 columns in the dataset. From now on, I will reference only this cleaned dataset.


## New Variables
New Variables were created such as County, relates to the physical address of the AirBnB Listing.  County is an important variable because there are several Councils in Co. Dublin:
[Dublin City Council, South Dublin County Council, Fingal County Council, Dun Laoghaire-rathdown County Council]
Cork has two different councils too: ['Cork City Council', 'Cork County Council']


Another new variable created was ['Province'] for Connacht, Munster, Leinster and Ulster. Unfortunately, this is only ROI data so the only counties from Ulster that are in the dataset are Donegal, Monaghan and Cavan.
Hence there are less AirBnBlistings for Ulster(ROI).

| Province | Count |
| ----------| ------------- |
| Leinster | 11316 |
|  Munster| 8069 |
| Connacht | 4849 |
| Ulster | 2344 |



## Exploring EDA

### What County and City Councils have the most AirBnB listing?

![council](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/council.png)

### What is the average price for listings in each City and County Council?



### What parishes have the most AirBnB listing?

![parish](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/parish.png)

### What is the average price for listings in each parishes?



### What counties have the most AirBnB listing?

![county](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county.png)

### What is the average price for listings in each county?



### What is the most common room type listed in Ireland?

![room_type](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/room_type.png)

There are a huge difference in the room type listing:


| Room Type        | Count         |
| ---------------- | ------------- |
| Entire home/apt  | 15433         |
| Private room     | 11139         |
| Hotel room       | 320           |
| Shared room      | 239           |


### What is the average price for listings in each room type?





### What are the most common words in the listings?
