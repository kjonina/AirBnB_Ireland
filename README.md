# AirBnB_Ireland
Conducting simple EDA and trying to predict AirBnB price 

# AirBnB Background
Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present a more unique, personalized way of experiencing the world. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Data analysis on millions of listings provided through Airbnb is a crucial factor for the company. These millions of listings generate a lot of data - data that can be analyzed and used for security, business decisions, understanding of customers' and providers' (hosts) behavior and performance on the platform, guiding marketing initiatives, implementation of innovative additional services and much more.


## Data Source

This dataset was collected from [AirBnb website](http://insideairbnb.com/get-the-data.html). It examines AirBnB listings for all of Ireland. This dataset has around 27131 observations in it with 16 columns and it is a mix between categorical and numeric values. 


## Business Questions:
The following Business Questions were thought of:

- [ ] What County and City Coucils have the most AirBnB listing?

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

![province_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/province_graph.png)
Please remember: Ulster only has Donegal, Monaghan and Cavan for Ulster.



## Exploring EDA

### What County and City Councils have the most AirBnB listing?

![neighbourhood_group_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/neighbourhood_group_graph.png)
This graph shows the number of AirBnB listings under each Council. Some councils may choose to limit the number of AirBnBs listings for rent control reasons. 


### What counties have the most AirBnB listing?

![county](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county.png)

### What is the average price for listings in each county?

![county_price_boxplot](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county_price_boxplot.png)

### What is the average  availabilty in each county?

![county_availability](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county_availability.png)


### What is the most common room type listed in Ireland?

![room_type](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/room_type.png)

There are a huge difference in the room type listing:


| Room Type        | Count         |
| ---------------- | ------------- |
| Entire home/apt  | 15349         |
| Private room     | 11119         |
| Hotel room       | 320           |
| Shared room      | 239           |


### What is the average price for listings in each room type?
![room_price_boxplot](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/room_price_boxplot.png)
The graph shows the price for each room type

### What is the average availablity for Each Room Type?
![room_availability_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/room_availability_graph.png)
The graph displays average availablity for each room type. 



### What are the most common words in the listings?
Not Finished

### What are some of the predictors of price?
Not Finished

# Personal Development
I came back to this dataset after about a month or two and I found it difficult to read my code or follow exactly where I left off. 
Note to self: comment better. 

I should also learn to do a loop to go through each county and place into a province. 
