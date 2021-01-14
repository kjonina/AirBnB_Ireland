# AirBnB_Ireland
Conducting simple EDA and trying to predict AirBnB price 

# AirBnB Background
Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present a more unique, personalized way of experiencing the world. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Data analysis on millions of listings provided through Airbnb is a crucial factor for the company. These millions of listings generate a lot of data - data that can be analyzed and used for security, business decisions, understanding of customers' and providers' (hosts) behavior and performance on the platform, guiding marketing initiatives, implementation of innovative additional services and much more.


## Data Source

This dataset was collected from [AirBnb website](http://insideairbnb.com/get-the-data.html). It examines AirBnB listings for all of Ireland. This dataset has around 27131 observations in it with 16 columns and it is a mix between categorical and numeric values. 


## Business Questions:
The following Business Questions were thought of:

- [x] What County and City Coucils have the most AirBnB listing?

- [x] What counties have the most AirBnB listing?
- [x] What counties have the highest average price?
- [x] What counties have the highest average availability?

- [x] What provinces have the most AirBnB listing?
- [x] What provinces have the highest average price?
- [x] What provinces have the highest average availability?

- [x] What is the most common room type listed in Ireland?
- [x] What is the average price for each room type?
- [x] What is the average availability for  each room type?

- [ ] What are the most common words in the listings?

- [ ] Create a model to predict price of an AirBnB listing.

# Learning Outcomes

The purpose of this analysis for myself is to: 
- [ ] deal with parts of code I struggle with: splitting text data, creating loops.
- [ ] deal with outliers in prices
- [ ] create a map (preferably a heatmap) for the locations of the AirBnB.
- [ ] create a more fancy looking graphs 
- [ ] run analysis on text column called ['Name'] 
- [ ] create a WordCloud for longitude['Name']
- [ ] create a predictive model for Prices 


## Data preparation
The following are variable 
['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
'neighbourhood', 'latitude', '', 'room_type', 'price',
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


![map_of_ROI](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/map_of_ROI.png) 

The graph shows the price of the AirBnB listings by location. 

## New Variables
New Variables were created such as County, relates to the physical address of the AirBnB Listing.  County is an important variable because there are several Councils in Co. Dublin:
[Dublin City Council, South Dublin County Council, Fingal County Council, Dun Laoghaire-rathdown County Council]
Cork has two different councils too: ['Cork City Council', 'Cork County Council']


Another new variable created was ['Province'] for Connacht, Munster, Leinster and Ulster. Unfortunately, this is only ROI data so the only counties from Ulster that are in the dataset are Donegal, Monaghan and Cavan.
Hence there are less AirBnBlistings for Ulster(ROI). =

| Province | Count |
| ----------| ------------- |
| Leinster | 11316 |
|  Munster| 8069 |
| Connacht | 4849 |
| Ulster | 2344 |

![province_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/province_graph.png)

Please remember: Ulster only has Donegal, Monaghan and Cavan for Ulster.

## Exploring EDA - County and City Councils

### What County and City Councils have the most AirBnB listing?

![neighbourhood_group_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/neighbourhood_group_graph.png)
This graph shows the number of AirBnB listings under each Council. Some councils may choose to limit the number of AirBnBs listings for rent control reasons. 


## Exploring EDA - Countries

### What counties have the most AirBnB listing?

![county_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county_graph.png)

### What is the average price for listings in each county?

![county_price_boxplot](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county_price_boxplot.png)

### What is the average  availabilty in each county?

![county_availability](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/county_availability_graph.png)
Dublin has the lowest average availability of 69 night, which is the lowest in the country. An explonation for this could be that the many of the AirBnBs are actually student accomodation for college students. When the students leave at the end of spring, the tenants put up their available rooms on AirBnB.


## Exploring EDA - Room Type

### What is the most common room type listed in Ireland?

![room_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/room_graph.png)

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

### Which hosts have the most listings?
![host_graph](https://github.com/kjonina/AirBnB_Ireland/blob/main/Graphs/host_graph.png)

There are some hosts that have over 100 listings on AirBnB! That is crazy!

So then I cheated and opened Tableau to look at what is happening in the dataset in relation to hosts. I found some very interesting details.
Here is my [Public Tableau Account](https://public.tableau.com/profile/karina.jonina#!/vizhome/AirBnBListingIreland/AirBnBListingsinIreland). Check out the dashboard and drill down by county or host.
It is a quick sketch to drill down and hence has no fancy-smancy stuff. 

### What are the most common words in the listings?
Not Finished

### What are some of the predictors of price?
Not Finished

# Personal Development
I came back to this dataset after about a month or two and I found it difficult to read my code or follow exactly where I left off. 
Note to self: comment better. 

I should also learn to do a loop to go through each county and place into a province. 

During my hiatus, I had to analyse a dataset for a college module where I used Python and Pandas too. Due to the volume of graphs I created, I learned how to automatically save graphs so that when I make adjustments to the graph for a billionth time, I don't have to go through the process of updating the graph in the folder. I take that as a win!
Now I use that tiny piece of code everywhere and it reduced my error with graphs. As long as I have the right name in this README.md, the updated graph will always be displayed!


# Learning Achieved
The purpose of this analysis for myself is to: 
- [x] deal with parts of code I struggle with: splitting text data, creating loops.
I did split data and created new variables which struggled with. The code is not the prettiest or the most concise but I just wanted to aimed for creating them no matter how limited my coding skills are.
Still have not created one loop. 

- [x] deal with outliers in prices
I dealt with the outlier in prices and used normalisation to eliminate any outliers.

- [x] create a map (preferably a heatmap) for the locations of the AirBnB.
I created *a* map for ROI which I was very pleased with myself over. 

- [x] create a more fancy looking graphs 
Well, I used a different tool, [ Tableau ](https://public.tableau.com/profile/karina.jonina#!/vizhome/AirBnBListingIreland/AirBnBListingsinIreland), to drill down on what is happening with host_id.
Not sure if this is cheating or just good thinking. I am sure it is possible to create in Python too. 
I also created three graphs side by side which I learned to hopefully aid better visuals and less writing.

# Post-Analysis Learning
My next challenge regarding this data set should be:
- [ ] Create loops for county and province variable creation.
- [ ] Created more maps 
