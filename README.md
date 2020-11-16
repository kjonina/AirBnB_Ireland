# AirBnB_Ireland
Conducting simple EDA and trying to predict AirBnB price 

# AirBnB Background
Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present a more unique, personalized way of experiencing the world. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Data analysis on millions of listings provided through Airbnb is a crucial factor for the company. These millions of listings generate a lot of data - data that can be analyzed and used for security, business decisions, understanding of customers' and providers' (hosts) behavior and performance on the platform, guiding marketing initiatives, implementation of innovative additional services and much more.


## Data Source

This dataset was collected from [AirBnb website](http://insideairbnb.com/get-the-data.html). It examines AirBnB listings for all of Ireland. This dataset has around 27131 observations in it with 16 columns and it is a mix between categorical and numeric values. 


## Business Questions:
The following Business Questions were thought of:

- [ ] What County and City Coucils have the most AirBnB listing?
- [ ] What is the average price for listings in each City and County Council?

- [ ] What parishes have the most AirBnB listing?
- [ ] What is the average price for listings in each parishes?

- [ ] What counties have the most AirBnB listing?
- [ ] What is the average price for listings in each county?

- [ ] What is the most common room type listed in Ireland?
- [ ] What is the average price for listings in each room type?

- [ ] What are the most common words in the listings?

- [ ] Create a model to predict price of an AirBnB listing.




# Learning Outcomes

The purpose of this analysis for myself is to: 
- [ ] deal with parts of code I struggle with: splitting text data, creating loops.
- [ ] create a more fancy looking graphs 
- [ ] run analysis on text column called ['Name'] 
- [ ] create a WordCloud for ['Name']
- [ ] create a predictive model for Prices 


## Variables
The following are variable 
['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
'minimum_nights', 'number_of_reviews', 'number_of_reviews_ltm',
'last_review', 'calculated_host_listings_count', 'availability_365'],


New Variables were created such as County.  County is an important variable because there are several Councils in Co. Dublin:
[Dublin City Council, South Dublin County Council, Fingal County Council, Dun Laoghaire-rathdown County Council]


## Exploring EDA

### What County and City Coucils have the most AirBnB listing?

![council](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/council.png)

### What is the average price for listings in each City and County Council?



### What parishes have the most AirBnB listing?

![parish](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/parish.png)

### What is the average price for listings in each parishes?



### What counties have the most AirBnB listing?

![county](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county.png)

### What is the average price for listings in each county?



### What is the most common room type listed in Ireland?**

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
