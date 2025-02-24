# Mahmoud Kefah
25/02/2025
Applied Data Science Capstone by IBM/Coursera
Table of contents
Introduction: Business Problem
Data
Methodology
Analysis
Results and Discussion
Conclusion
Introduction: Business Problem
In this project we will try to find the best apartment for rent due to a job offer. This project is targeted to everyone that might be interested to know more about rent prices of apartments, statistics of the neighborhoods or just want to move to São Paulo, Brazil.

Since our job offer it is for a specific address, we have selected a few close neighborhoods to live in. Although we have selected these neighborhoods, still there are a lot of apartments for rent. We have a few exigences regarding the nearby venues, and that will be taken into account to choose our home.

We will use our techniques in data science to generate the most promissing apartments for us to live in. Economic data and social characteristics of the neighborhoods will be considered in the proccess.

Data
Based on definition of our problem, factors that will influence our decision are:

the apartment must have at least 2 bedrooms
the monthly value of the rent cannot exceed R$ 1.700,00
it has to be at least 5km from workplace
it has to be near at least 1km from a gym and a market
social and economic characteristics of the neighborhod based on the available data
We decided to select a few neighborhoods around our workplace to extract our data.

Following data sources will be needed to extract/generate the required information:

coordinate of workplace will be obtained using geocoder
economical and location data of rental properties around the workplace will be obtained through webscraping of a major online portal.
nearby venues around each neighborhood will be obtained using Foursquare API
Let's load our libraries and our apartments data
import pandas as pd
import numpy as np
import json
import geocoder
from haversine import haversine
import requests
import folium
from sklearn.cluster import KMeans
# Before we start, let's just define out basic factors.
starting_point = 'Rua Girassol, 555, Vila Madalena, São Paulo/SP'
max_rent_value = 1700
min_bedrooms = 2
min_dist_workplace = 5
min_dist_gym = 1
min_dist_mkt = 1
dataset = pd.read_csv('rent_properties_data.csv')
dataset.head()
code	type	neighborhood	rent_value	condominium_value	property_tax	bedrooms	suites	bathrooms	parking_spaces	private_area	furnished	amenities	latitude	longitude
0	0	APARTMENT	Perdizes	2600	1000.0	NaN	2	-	2	1	98	False	['PETS_ALLOWED', 'CONCIERGE_24H']	-23.533013	-46.664833
1	1	APARTMENT	Higienópolis	2111	689.0	NaN	2	-	1	-	62	False	['ELEVATOR', 'PETS_ALLOWED', 'BICYCLES_PLACE',...	-23.543967	-46.651519
2	2	APARTMENT	Liberdade	2800	570.0	2886.0	3	1	2	2	79	False	['POOL', 'GYM', 'BARBECUE_GRILL', 'GATED_COMMU...	-23.560154	-46.634183
3	3	APARTMENT	Vila Madalena	1900	411.0	85.0	2	-	1	1	48	False	['PETS_ALLOWED', 'BICYCLES_PLACE', 'ELECTRONIC...	-23.554346	-46.695942
4	4	APARTMENT	Higienópolis	3000	900.0	380.0	3	-	2	1	127	False	['GYM', 'BARBECUE_GRILL', 'GATED_COMMUNITY', '...	-23.548102	-46.654299
dataset.shape
(3081, 15)
Let's just reassure that all of the properties are apartments.

dataset['type'].value_counts()
APARTMENT    3081
Name: type, dtype: int64
Now that we know that we only have apartments, we can drop this column.

dataset.drop(columns=['type'], inplace=True)
Now, let's get our workplace coordinates in order to get the nearby venues.

work_location = geocoder.arcgis(starting_point).latlng
work_location
[-23.554040008948206, -46.69029000526774]
dataset.head()
code	neighborhood	rent_value	condominium_value	property_tax	bedrooms	suites	bathrooms	parking_spaces	private_area	furnished	amenities	latitude	longitude
0	0	Perdizes	2600	1000.0	NaN	2	-	2	1	98	False	['PETS_ALLOWED', 'CONCIERGE_24H']	-23.533013	-46.664833
1	1	Higienópolis	2111	689.0	NaN	2	-	1	-	62	False	['ELEVATOR', 'PETS_ALLOWED', 'BICYCLES_PLACE',...	-23.543967	-46.651519
2	2	Liberdade	2800	570.0	2886.0	3	1	2	2	79	False	['POOL', 'GYM', 'BARBECUE_GRILL', 'GATED_COMMU...	-23.560154	-46.634183
3	3	Vila Madalena	1900	411.0	85.0	2	-	1	1	48	False	['PETS_ALLOWED', 'BICYCLES_PLACE', 'ELECTRONIC...	-23.554346	-46.695942
4	4	Higienópolis	3000	900.0	380.0	3	-	2	1	127	False	['GYM', 'BARBECUE_GRILL', 'GATED_COMMUNITY', '...	-23.548102	-46.654299
Let's reassure that every latitude and longitude values are filled.

dataset = dataset.loc[(dataset['latitude'] != '-') & (dataset['longitude'] != '-')]
dataset.shape
(3081, 14)
Now we have to measure the distance between each apartment to our workplace. First, let's create a function to merge latitude and longitude in a single row.

def get_location(dataframe, name_latitude, name_longitude):
    df = dataframe.copy()
    df['location'] = df[name_latitude].astype(str) + ',' + df[name_longitude].astype(str)
    df['location'] = df['location'].apply(lambda x: x.split(','))
    df['location'] = df['location'].apply(lambda x: [float(i) for i in x])
    return df
dataset = get_location(dataset, name_latitude='latitude', name_longitude='longitude')
dataset.head()
code	neighborhood	rent_value	condominium_value	property_tax	bedrooms	suites	bathrooms	parking_spaces	private_area	furnished	amenities	latitude	longitude	location
0	0	Perdizes	2600	1000.0	NaN	2	-	2	1	98	False	['PETS_ALLOWED', 'CONCIERGE_24H']	-23.533013	-46.664833	[-23.533013, -46.664833]
1	1	Higienópolis	2111	689.0	NaN	2	-	1	-	62	False	['ELEVATOR', 'PETS_ALLOWED', 'BICYCLES_PLACE',...	-23.543967	-46.651519	[-23.543967, -46.651519]
2	2	Liberdade	2800	570.0	2886.0	3	1	2	2	79	False	['POOL', 'GYM', 'BARBECUE_GRILL', 'GATED_COMMU...	-23.560154	-46.634183	[-23.560154, -46.634183]
3	3	Vila Madalena	1900	411.0	85.0	2	-	1	1	48	False	['PETS_ALLOWED', 'BICYCLES_PLACE', 'ELECTRONIC...	-23.554346	-46.695942	[-23.554346, -46.695942]
4	4	Higienópolis	3000	900.0	380.0	3	-	2	1	127	False	['GYM', 'BARBECUE_GRILL', 'GATED_COMMUNITY', '...	-23.548102	-46.654299	[-23.548102, -46.654299]
Let's use the Haversine library to measure the distance between points using the latitude and longitude of them. The distance it's given in km.

dataset['distance_from_workplace'] = dataset['location'].apply(lambda x: haversine(work_location, x))
dataset.head()
code	neighborhood	rent_value	condominium_value	property_tax	bedrooms	suites	bathrooms	parking_spaces	private_area	furnished	amenities	latitude	longitude	location	distance_from_workplace
0	0	Perdizes	2600	1000.0	NaN	2	-	2	1	98	False	['PETS_ALLOWED', 'CONCIERGE_24H']	-23.533013	-46.664833	[-23.533013, -46.664833]	3.492998
1	1	Higienópolis	2111	689.0	NaN	2	-	1	-	62	False	['ELEVATOR', 'PETS_ALLOWED', 'BICYCLES_PLACE',...	-23.543967	-46.651519	[-23.543967, -46.651519]	4.107762
2	2	Liberdade	2800	570.0	2886.0	3	1	2	2	79	False	['POOL', 'GYM', 'BARBECUE_GRILL', 'GATED_COMMU...	-23.560154	-46.634183	[-23.560154, -46.634183]	5.759161
3	3	Vila Madalena	1900	411.0	85.0	2	-	1	1	48	False	['PETS_ALLOWED', 'BICYCLES_PLACE', 'ELECTRONIC...	-23.554346	-46.695942	[-23.554346, -46.695942]	0.577115
4	4	Higienópolis	3000	900.0	380.0	3	-	2	1	127	False	['GYM', 'BARBECUE_GRILL', 'GATED_COMMUNITY', '...	-23.548102	-46.654299	[-23.548102, -46.654299]	3.727615
Now, let's establish a 5km radius of our workplace and filter all apartments inside of it.

dataset = dataset.loc[dataset['distance_from_workplace'] <= min_dist_workplace]
dataset.shape
(2906, 16)
dataset.head()
code	neighborhood	rent_value	condominium_value	property_tax	bedrooms	suites	bathrooms	parking_spaces	private_area	furnished	amenities	latitude	longitude	location	distance_from_workplace
0	0	Perdizes	2600	1000.0	NaN	2	-	2	1	98	False	['PETS_ALLOWED', 'CONCIERGE_24H']	-23.533013	-46.664833	[-23.533013, -46.664833]	3.492998
1	1	Higienópolis	2111	689.0	NaN	2	-	1	-	62	False	['ELEVATOR', 'PETS_ALLOWED', 'BICYCLES_PLACE',...	-23.543967	-46.651519	[-23.543967, -46.651519]	4.107762
3	3	Vila Madalena	1900	411.0	85.0	2	-	1	1	48	False	['PETS_ALLOWED', 'BICYCLES_PLACE', 'ELECTRONIC...	-23.554346	-46.695942	[-23.554346, -46.695942]	0.577115
4	4	Higienópolis	3000	900.0	380.0	3	-	2	1	127	False	['GYM', 'BARBECUE_GRILL', 'GATED_COMMUNITY', '...	-23.548102	-46.654299	[-23.548102, -46.654299]	3.727615
5	5	Vila Pompéia	2150	750.0	122.0	2	-	2	1	56	False	['POOL', 'GYM', 'ELEVATOR', 'PETS_ALLOWED', 'P...	-23.528687	-46.685394	[-23.528687, -46.685394]	2.862970
Now that we have our dataset cleaned, let's create a map to visualize the location of every apartment in a 5km radius of our workplace.

# instantiate the map
map_city = folium.Map(location=[work_location[0], work_location[1]], zoom_start=14)

# add the markers of every apartment
for lat, lng, label, distance in zip(dataset['latitude'], dataset['longitude'], dataset['code'], dataset['distance_from_workplace']):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=f'code={label}\ndistance={round(distance, 2)}km',
        fill=True,
        color='blue',
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(map_city)

# add the workplace marker
folium.Marker(
    [work_location[0], work_location[1]],
    radius=5,
    tooltip='Workplace',
    icon=folium.Icon(icon='briefcase', color="green")
).add_to(map_city)

# add the 5km radius
folium.Circle([work_location[0], work_location[1]],
                    radius=5000,
                    color='red'
                   ).add_to(map_city)

map_city
Make this Notebook Trusted to load map: File -> Trust Notebook
Now, just to complement our visualization, let's create a heatmap to see how our apartments are distributed.

from folium import plugins
from folium.plugins import HeatMap

heatmap_city = folium.Map(location=[work_location[0], work_location[1]], zoom_start=14)

folium.TileLayer('cartodbpositron').add_to(heatmap_city) #cartodbpositron cartodbdark_matter
HeatMap([[res[12], res[13]] for res in dataset.values]).add_to(heatmap_city)

# add the workplace marker
folium.Marker(
    [work_location[0], work_location[1]],
    radius=5,
    tooltip='Workplace',
    icon=folium.Icon(icon='briefcase', color="green")
).add_to(heatmap_city)

# add the 5km radius
folium.Circle([work_location[0], work_location[1]],
                    radius=5000,
                    color='red'
                   ).add_to(heatmap_city)

heatmap_city
Make this Notebook Trusted to load map: File -> Trust Notebook
Now that we have the properties mapped, let's extract the nearby venues through every neighborhood.

Foursquare
Let's fill our credentials to use the foursquare API.

CLIENT_ID = '' # your Foursquare ID
CLIENT_SECRET = '' # your Foursquare Secret
VERSION = '' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
Your credentails:
CLIENT_ID: 
CLIENT_SECRET:
Now let's see which are the neighborhoods to extract the nearby venues.

dataset['neighborhood'].value_counts()
Pinheiros                878
Perdizes                 811
Higienópolis             263
Vila Pompéia             235
Santa Cecília            172
Vila Madalena            146
Sumarezinho              115
Cerqueira César           64
Consolação                53
Sumaré                    35
Água Branca               32
Vila Anglo Brasileira     31
Vila Buarque              26
Pompeia                    9
Jardim das Bandeiras       8
Barra Funda                3
Bela Vista                 3
Liberdade                  3
Alto de Pinheiros          3
Jardim Vera Cruz           2
Vila Romana                2
Jardim Paineiras           2
Várzea da Barra Funda      2
Pacaembu                   2
Itaim Bibi                 1
Morada do Sol              1
Cerq Cesar                 1
Jardim Paulistano          1
Santa Cecilia              1
Brás                       1
Name: neighborhood, dtype: int64
We can see that there are some neighborhoods that are the same but with different names, let's normalize this.

dataset.loc[dataset['neighborhood'] == 'Santa Cecilia', 'neighborhood'] = 'Santa Cecília'
dataset.loc[dataset['neighborhood'] == 'Pompeia', 'neighborhood'] = 'Vila Pompéia'
dataset.loc[dataset['neighborhood'] == 'Cerq Cesar', 'neighborhood'] = 'Cerqueira César'
dataset.loc[dataset['neighborhood'] == 'Várzea da Barra Funda', 'neighborhood'] = 'Barra Funda'
dataset.loc[dataset['neighborhood'] == 'Alto de Pinheiros', 'neighborhood'] = 'Pinheiros'
dataset['neighborhood'].value_counts()
Pinheiros                881
Perdizes                 811
Higienópolis             263
Vila Pompéia             244
Santa Cecília            173
Vila Madalena            146
Sumarezinho              115
Cerqueira César           65
Consolação                53
Sumaré                    35
Água Branca               32
Vila Anglo Brasileira     31
Vila Buarque              26
Jardim das Bandeiras       8
Barra Funda                5
Liberdade                  3
Bela Vista                 3
Jardim Vera Cruz           2
Jardim Paineiras           2
Vila Romana                2
Pacaembu                   2
Itaim Bibi                 1
Jardim Paulistano          1
Morada do Sol              1
Brás                       1
Name: neighborhood, dtype: int64
Ok, now let's use the geocoder library to extract the coordinates of every neighborhood.

neighborhood_data = pd.DataFrame(columns=['Neighborhood', 'Latitude', 'Longitude'])

for bairro in dataset['neighborhood'].unique():
    location = geocoder.arcgis(bairro + ', São Paulo/SP').latlng
    neighborhood_data = neighborhood_data.append({'Neighborhood': bairro, 'Latitude': location[0], 'Longitude': location[1]}, ignore_index=True)
neighborhood_data
Neighborhood	Latitude	Longitude
0	Perdizes	-23.54057	-46.67236
1	Higienópolis	-23.54523	-46.65975
2	Vila Madalena	-23.55119	-46.69711
3	Vila Pompéia	-23.53189	-46.68627
4	Pinheiros	-23.56200	-46.68597
5	Cerqueira César	-23.56218	-46.66445
6	Santa Cecília	-23.53592	-46.65775
7	Água Branca	-23.51530	-46.69324
8	Sumarezinho	-23.54625	-46.69408
9	Consolação	-23.55520	-46.65713
10	Jardim das Bandeiras	-23.55262	-46.68552
11	Vila Romana	-23.52922	-46.69692
12	Vila Anglo Brasileira	-23.53909	-46.69351
13	Vila Buarque	-23.54327	-46.64888
14	Jardim Paineiras	-23.26208	-47.28539
15	Sumaré	-23.54298	-46.68546
16	Jardim Vera Cruz	-23.61757	-46.47371
17	Jardim Paulistano	-23.57191	-46.68685
18	Bela Vista	-23.56001	-46.64523
19	Liberdade	-23.55839	-46.63292
20	Morada do Sol	-23.55953	-46.67948
21	Pacaembu	-23.54305	-46.66613
22	Barra Funda	-23.52910	-46.66208
23	Itaim Bibi	-23.58670	-46.67828
24	Brás	-23.54277	-46.62000
Let's create a function to get the category type of each venue.

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
And finally let's create a function to iterate through every neighborhood, extract the nearby venues and put that into a dataframe.

# define a limit for each neighborhood
LIMIT = 100

#define the function and set the radius to 1km
def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
# type your answer here
dataset_venues = getNearbyVenues(names=neighborhood_data['Neighborhood'],
                                 latitudes=neighborhood_data['Latitude'],
                                 longitudes=neighborhood_data['Longitude']
                                  )
dataset_venues.head()
Neighborhood	Neighborhood Latitude	Neighborhood Longitude	Venue	Venue Latitude	Venue Longitude	Venue Category
0	Perdizes	-23.54057	-46.67236	Droga Raia	-23.539772	-46.675214	Pharmacy
1	Perdizes	-23.54057	-46.67236	Ofner	-23.541195	-46.670611	Dessert Shop
2	Perdizes	-23.54057	-46.67236	Kaori Sushi	-23.542104	-46.671712	Japanese Restaurant
3	Perdizes	-23.54057	-46.67236	Atelier de La Musique	-23.537454	-46.673190	Music School
4	Perdizes	-23.54057	-46.67236	Zaccara Livraria	-23.541946	-46.671583	Bookstore
dataset_venues.shape
(2354, 7)
Let's check how many venues were returned for each neighborhood

dataset_venues.groupby('Neighborhood').count()
Neighborhood Latitude	Neighborhood Longitude	Venue	Venue Latitude	Venue Longitude	Venue Category
Neighborhood						
Barra Funda	64	64	64	64	64	64
Bela Vista	100	100	100	100	100	100
Brás	100	100	100	100	100	100
Cerqueira César	100	100	100	100	100	100
Consolação	100	100	100	100	100	100
Higienópolis	100	100	100	100	100	100
Itaim Bibi	100	100	100	100	100	100
Jardim Paineiras	69	69	69	69	69	69
Jardim Paulistano	100	100	100	100	100	100
Jardim Vera Cruz	36	36	36	36	36	36
Jardim das Bandeiras	100	100	100	100	100	100
Liberdade	100	100	100	100	100	100
Morada do Sol	100	100	100	100	100	100
Pacaembu	100	100	100	100	100	100
Perdizes	100	100	100	100	100	100
Pinheiros	100	100	100	100	100	100
Santa Cecília	100	100	100	100	100	100
Sumarezinho	100	100	100	100	100	100
Sumaré	100	100	100	100	100	100
Vila Anglo Brasileira	100	100	100	100	100	100
Vila Buarque	100	100	100	100	100	100
Vila Madalena	100	100	100	100	100	100
Vila Pompéia	100	100	100	100	100	100
Vila Romana	100	100	100	100	100	100
Água Branca	85	85	85	85	85	85
Let's find out how many unique categories can be curated from all the returned venues

print(f'There are {dataset_venues["Venue Category"].nunique()} uniques categories.')
There are 246 uniques categories.
Now that we have our venues, let's filter the ones that we are interested at: gyms, markets and metros.

gym_venues = dataset_venues.loc[dataset_venues['Venue Category'].str.contains('Gym')]
print(f'There are {gym_venues.shape[0]} gym venues')
There are 107 gym venues
market_venues = dataset_venues.loc[dataset_venues['Venue Category'].str.contains('Market')]
print(f'There are {market_venues.shape[0]} market venues')
There are 42 market venues
To finalize the data section, let's add to our previous map every venue in the above categories.

# Add markets and gyms to the map

for lat, lng, label in zip(gym_venues['Venue Latitude'], gym_venues['Venue Longitude'], gym_venues['Venue']):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=f'{label}',
        fill=True,
        color='green',
        fill_color='green',
        fill_opacity=0.6
).add_to(map_city)

for lat, lng, label in zip(market_venues['Venue Latitude'], market_venues['Venue Longitude'], market_venues['Venue']):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=f'{label}',
        fill=True,
        color='red',
        fill_color='red',
        fill_opacity=0.6
).add_to(map_city)

map_city
Make this Notebook Trusted to load map: File -> Trust Notebook
Methodology
In this project we have to find an apartment that fit our needs.

In first step we have collected the required data: rental properties data, type (category) of every venue in a 1km radius from each neighborhood. We also have filtered every apartment in a 5km radius from our workplace.

In the second step we will be performing a exploratory data analysis in order to extract information about economic and social data of every neighborhood and perform a k-means to cluster the neighborhoods.

We will calculate the distance through every selected venue (gyms and markets) to every apartment, in order to find out the nearest venue of each category to every apartment.

Finally we will filter our dataset to fit our basic needs regarded as value of the rent, distance to the specific venues and number of bedrooms.

Analysis
Let's first analyze each neighborhood and find out more about it each one.

# one hot encoding
neighborhood_onehot = pd.get_dummies(dataset_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
neighborhood_onehot['Neighborhood'] = dataset_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [neighborhood_onehot.columns[-1]] + list(neighborhood_onehot.columns[:-1])
neighborhood_onehot = neighborhood_onehot[fixed_columns]

neighborhood_onehot.head()
Neighborhood	Acai House	Accessories Store	American Restaurant	Antique Shop	Argentinian Restaurant	Art Gallery	Art Museum	Arts & Crafts Store	Arts & Entertainment	...	Train Station	Travel Agency	University	Vegetarian / Vegan Restaurant	Vietnamese Restaurant	Warehouse Store	Wine Bar	Wine Shop	Women's Store	Yoga Studio
0	Perdizes	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	Perdizes	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	Perdizes	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	Perdizes	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	Perdizes	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 247 columns

neighborhood_onehot.shape
(2354, 247)
Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

neighborhood_grouped = neighborhood_onehot.groupby('Neighborhood').mean().reset_index()
neighborhood_grouped
Neighborhood	Acai House	Accessories Store	American Restaurant	Antique Shop	Argentinian Restaurant	Art Gallery	Art Museum	Arts & Crafts Store	Arts & Entertainment	...	Train Station	Travel Agency	University	Vegetarian / Vegan Restaurant	Vietnamese Restaurant	Warehouse Store	Wine Bar	Wine Shop	Women's Store	Yoga Studio
0	Barra Funda	0.00	0.00	0.00	0.00	0.00	0.00	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.000000
1	Bela Vista	0.00	0.00	0.00	0.01	0.00	0.01	0.000000	0.01	0.00	...	0.000000	0.000000	0.00	0.00	0.01	0.00	0.00	0.01	0.00	0.000000
2	Brás	0.00	0.01	0.00	0.00	0.00	0.00	0.000000	0.01	0.00	...	0.010000	0.000000	0.00	0.00	0.00	0.01	0.01	0.01	0.00	0.000000
3	Cerqueira César	0.00	0.00	0.00	0.00	0.02	0.00	0.000000	0.01	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.01	0.000000
4	Consolação	0.00	0.00	0.00	0.00	0.00	0.01	0.010000	0.00	0.00	...	0.000000	0.000000	0.00	0.01	0.00	0.00	0.00	0.00	0.00	0.000000
5	Higienópolis	0.00	0.00	0.00	0.00	0.00	0.01	0.010000	0.00	0.00	...	0.000000	0.000000	0.00	0.01	0.00	0.00	0.00	0.00	0.00	0.010000
6	Itaim Bibi	0.00	0.00	0.00	0.00	0.02	0.00	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.020000
7	Jardim Paineiras	0.00	0.00	0.00	0.00	0.00	0.00	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.014493
8	Jardim Paulistano	0.00	0.00	0.00	0.00	0.00	0.02	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.02	0.00	0.00	0.01	0.01	0.00	0.010000
9	Jardim Vera Cruz	0.00	0.00	0.00	0.00	0.00	0.00	0.000000	0.00	0.00	...	0.000000	0.027778	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.000000
10	Jardim das Bandeiras	0.00	0.00	0.00	0.00	0.01	0.07	0.000000	0.01	0.00	...	0.000000	0.000000	0.01	0.02	0.00	0.00	0.00	0.00	0.01	0.010000
11	Liberdade	0.00	0.00	0.01	0.00	0.00	0.00	0.000000	0.01	0.00	...	0.000000	0.000000	0.00	0.02	0.00	0.00	0.00	0.00	0.00	0.000000
12	Morada do Sol	0.00	0.00	0.00	0.00	0.01	0.03	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.02	0.00	0.00	0.00	0.01	0.01	0.000000
13	Pacaembu	0.00	0.00	0.00	0.00	0.02	0.01	0.020000	0.00	0.00	...	0.000000	0.000000	0.00	0.02	0.00	0.00	0.00	0.00	0.00	0.010000
14	Perdizes	0.00	0.00	0.00	0.00	0.02	0.00	0.010000	0.00	0.02	...	0.000000	0.000000	0.00	0.02	0.00	0.00	0.00	0.00	0.00	0.010000
15	Pinheiros	0.00	0.00	0.00	0.00	0.01	0.05	0.000000	0.02	0.00	...	0.000000	0.000000	0.00	0.02	0.00	0.00	0.01	0.00	0.01	0.020000
16	Santa Cecília	0.00	0.00	0.00	0.00	0.00	0.02	0.000000	0.01	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.010000
17	Sumarezinho	0.01	0.00	0.00	0.00	0.00	0.02	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.01	0.010000
18	Sumaré	0.00	0.00	0.00	0.00	0.00	0.01	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.01	0.00	0.00	0.00	0.00	0.00	0.010000
19	Vila Anglo Brasileira	0.00	0.00	0.00	0.00	0.00	0.01	0.000000	0.01	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.010000
20	Vila Buarque	0.00	0.00	0.00	0.00	0.00	0.01	0.000000	0.02	0.00	...	0.000000	0.000000	0.00	0.01	0.00	0.00	0.00	0.00	0.00	0.000000
21	Vila Madalena	0.01	0.00	0.00	0.00	0.00	0.03	0.000000	0.00	0.00	...	0.000000	0.000000	0.01	0.00	0.00	0.00	0.00	0.00	0.01	0.000000
22	Vila Pompéia	0.00	0.00	0.00	0.00	0.00	0.00	0.000000	0.01	0.00	...	0.000000	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.020000
23	Vila Romana	0.00	0.00	0.00	0.00	0.00	0.00	0.000000	0.00	0.00	...	0.000000	0.000000	0.00	0.01	0.00	0.00	0.00	0.01	0.00	0.010000
24	Água Branca	0.00	0.00	0.00	0.00	0.00	0.00	0.011765	0.00	0.00	...	0.011765	0.000000	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.000000
25 rows × 247 columns

print(f'We have {neighborhood_grouped.shape[0]} neighborhoods and {neighborhood_grouped.shape[1]} venues category')
We have 25 neighborhoods and 247 venues category
Now let's create a new dataframe and display the top 10 venues for each neighborhood to find out more about the social characteristics.

First, let's write a function to sort the venues in descending order.

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
And now, create the dataframe

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = neighborhood_grouped['Neighborhood']

for ind in np.arange(neighborhood_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(neighborhood_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
Neighborhood	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
0	Barra Funda	Brazilian Restaurant	Pizza Place	Restaurant	Café	Dessert Shop	Italian Restaurant	Supermarket	Beer Bar	Farmers Market	Food Truck
1	Bela Vista	Theater	Bakery	Japanese Restaurant	Bar	Italian Restaurant	Cosmetics Shop	Pizza Place	Brazilian Restaurant	Café	Coffee Shop
2	Brás	Brazilian Restaurant	Clothing Store	Sandwich Place	Italian Restaurant	Food & Drink Shop	Market	Restaurant	Deli / Bodega	Hotel	Health Food Store
3	Cerqueira César	Italian Restaurant	Hotel	Brazilian Restaurant	Ice Cream Shop	French Restaurant	Restaurant	Gym / Fitness Center	Seafood Restaurant	Bookstore	Pizza Place
4	Consolação	Brazilian Restaurant	Gym / Fitness Center	Coffee Shop	Theater	Bakery	Gym	Movie Theater	Bar	Burger Joint	Dance Studio
Now, let's group the economic data of each neighborhood. First let's get the size of each neighborhood to know if it have enough registers to be taken into account.

neighborhood_economic = dataset.groupby('neighborhood').mean()[['rent_value', 'condominium_value', 'property_tax', 'bedrooms', 'bathrooms', 'private_area']].reset_index()
neighborhood_economic['size'] = dataset.groupby('neighborhood').size().values
neighborhood_economic.head()
neighborhood	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size
0	Barra Funda	3272.600000	462.600000	107.333333	2.200000	1.600000	88.400000	5
1	Bela Vista	3066.666667	973.333333	307.666667	3.000000	2.000000	85.000000	3
2	Brás	4000.000000	1700.000000	100.000000	3.000000	1.000000	168.000000	1
3	Cerqueira César	3552.523077	1139.140625	299.377049	2.353846	2.169231	98.000000	65
4	Consolação	78372.735849	2149.943396	695.442308	2.905660	2.622642	157.754717	53
Finally, let's merge the economical and the social characteristics of the neighborhoods.

neighborhood_social_economic = neighborhood_economic.merge(neighborhoods_venues_sorted, left_on='neighborhood', right_on='Neighborhood').drop(columns=['Neighborhood'])
neighborhood_social_economic.head()
neighborhood	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
0	Barra Funda	3272.600000	462.600000	107.333333	2.200000	1.600000	88.400000	5	Brazilian Restaurant	Pizza Place	Restaurant	Café	Dessert Shop	Italian Restaurant	Supermarket	Beer Bar	Farmers Market	Food Truck
1	Bela Vista	3066.666667	973.333333	307.666667	3.000000	2.000000	85.000000	3	Theater	Bakery	Japanese Restaurant	Bar	Italian Restaurant	Cosmetics Shop	Pizza Place	Brazilian Restaurant	Café	Coffee Shop
2	Brás	4000.000000	1700.000000	100.000000	3.000000	1.000000	168.000000	1	Brazilian Restaurant	Clothing Store	Sandwich Place	Italian Restaurant	Food & Drink Shop	Market	Restaurant	Deli / Bodega	Hotel	Health Food Store
3	Cerqueira César	3552.523077	1139.140625	299.377049	2.353846	2.169231	98.000000	65	Italian Restaurant	Hotel	Brazilian Restaurant	Ice Cream Shop	French Restaurant	Restaurant	Gym / Fitness Center	Seafood Restaurant	Bookstore	Pizza Place
4	Consolação	78372.735849	2149.943396	695.442308	2.905660	2.622642	157.754717	53	Brazilian Restaurant	Gym / Fitness Center	Coffee Shop	Theater	Bakery	Gym	Movie Theater	Bar	Burger Joint	Dance Studio
Now, let's run a k-means to cluster our neighborhoods
Let's merge the neighborhood_grouped dataset to our economic dataset. Since k-means can't take null values, let's fill them and assume that it is 0.

neighborhood_grouped = neighborhood_grouped.merge(neighborhood_economic, left_on='Neighborhood', right_on='neighborhood').drop(columns=['neighborhood']).fillna(0)
neighborhood_grouped.head()
Neighborhood	Acai House	Accessories Store	American Restaurant	Antique Shop	Argentinian Restaurant	Art Gallery	Art Museum	Arts & Crafts Store	Arts & Entertainment	...	Wine Shop	Women's Store	Yoga Studio	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size
0	Barra Funda	0.0	0.00	0.0	0.00	0.00	0.00	0.00	0.00	0.0	...	0.00	0.00	0.0	3272.600000	462.600000	107.333333	2.200000	1.600000	88.400000	5
1	Bela Vista	0.0	0.00	0.0	0.01	0.00	0.01	0.00	0.01	0.0	...	0.01	0.00	0.0	3066.666667	973.333333	307.666667	3.000000	2.000000	85.000000	3
2	Brás	0.0	0.01	0.0	0.00	0.00	0.00	0.00	0.01	0.0	...	0.01	0.00	0.0	4000.000000	1700.000000	100.000000	3.000000	1.000000	168.000000	1
3	Cerqueira César	0.0	0.00	0.0	0.00	0.02	0.00	0.00	0.01	0.0	...	0.00	0.01	0.0	3552.523077	1139.140625	299.377049	2.353846	2.169231	98.000000	65
4	Consolação	0.0	0.00	0.0	0.00	0.00	0.01	0.01	0.00	0.0	...	0.00	0.00	0.0	78372.735849	2149.943396	695.442308	2.905660	2.622642	157.754717	53
5 rows × 254 columns

Run k-means to cluster the neighborhood into 5 clusters.

# set number of clusters
kclusters = 5

neighborhood_grouped_clustering = neighborhood_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(neighborhood_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
array([0, 0, 0, 0, 2, 3, 0, 0, 0, 0])
Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# add clustering labels
neighborhood_social_economic.insert(0, 'Cluster Labels', kmeans.labels_)

neighborhood_merged = neighborhood_data

# merge neighborhood_data with neighborhood_social_economic to add latitude/longitude for each neighborhood
neighborhood_merged = neighborhood_merged.merge(neighborhood_social_economic, left_on='Neighborhood', right_on='neighborhood').drop(columns=['neighborhood'])

neighborhood_merged.head()
Neighborhood	Latitude	Longitude	Cluster Labels	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	...	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
0	Perdizes	-23.54057	-46.67236	1	44128.567201	1335.612422	410.262484	2.655980	2.366215	105.293465	...	Dessert Shop	Pet Store	Pharmacy	Bar	Spa	Burger Joint	Martial Arts School	Athletics & Sports	Pizza Place	Ice Cream Shop
1	Higienópolis	-23.54523	-46.65975	3	30550.163498	3905.482890	400.652174	2.631179	2.334601	121.076046	...	Bakery	Athletics & Sports	Coffee Shop	Italian Restaurant	Pizza Place	Middle Eastern Restaurant	Plaza	Dessert Shop	Gym / Fitness Center	Spa
2	Vila Madalena	-23.55119	-46.69711	3	20661.958904	1469.150685	232.496403	2.260274	1.917808	79.260274	...	Bar	Hostel	Brazilian Restaurant	Pizza Place	Burger Joint	Plaza	Bistro	Restaurant	Ice Cream Shop	Art Gallery
3	Vila Pompéia	-23.53189	-46.68627	4	58388.938525	963.443983	529.966387	2.557377	2.139344	81.106557	...	Bar	Italian Restaurant	Pizza Place	Brazilian Restaurant	Ice Cream Shop	Burger Joint	Farmers Market	Coffee Shop	Yoga Studio	Music Venue
4	Pinheiros	-23.56200	-46.68597	3	14737.455165	1025.800683	328.864571	2.293984	1.849035	84.118048	...	Art Gallery	Pet Store	Italian Restaurant	Bookstore	Ice Cream Shop	Burger Joint	Café	Bar	Clothing Store	Cocktail Bar
5 rows × 21 columns

Finally, let's visualize the resulting clusters

# create map
map_clusters = folium.Map(location=[work_location[0], work_location[1]], zoom_start=14)

colors=['red', 'yellow', 'orange', 'green', 'blue']

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(neighborhood_merged['Latitude'], neighborhood_merged['Longitude'], neighborhood_merged['Neighborhood'], neighborhood_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=colors[cluster-1],
        fill=True,
        fill_color=colors[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
Make this Notebook Trusted to load map: File -> Trust Notebook
Let's see how much registers there are in each cluster.

for cluster in np.sort(neighborhood_merged['Cluster Labels'].unique()):
    print(f'There are {neighborhood_merged.loc[neighborhood_merged["Cluster Labels"] == cluster].shape[0]} registers on Cluster {cluster}')
There are 16 registers on Cluster 0
There are 2 registers on Cluster 1
There are 1 registers on Cluster 2
There are 5 registers on Cluster 3
There are 1 registers on Cluster 4
Now, let's examine each cluster to find out more about it.

Cluster 1
neighborhood_merged.loc[neighborhood_merged['Cluster Labels'] == 0, neighborhood_merged.columns[[0] + list(range(4, neighborhood_merged.shape[1]))]]
Neighborhood	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
5	Cerqueira César	3552.523077	1139.140625	299.377049	2.353846	2.169231	98.000000	65	Italian Restaurant	Hotel	Brazilian Restaurant	Ice Cream Shop	French Restaurant	Restaurant	Gym / Fitness Center	Seafood Restaurant	Bookstore	Pizza Place
8	Sumarezinho	3673.695652	1088.878261	213.545455	2.252174	2.078261	84.834783	115	Hostel	Brazilian Restaurant	Bar	Gym	Plaza	Pizza Place	Pharmacy	Ice Cream Shop	Burger Joint	Park
10	Jardim das Bandeiras	3413.125000	1262.000000	295.625000	2.250000	2.000000	85.125000	8	Bar	Art Gallery	Hostel	Pizza Place	Plaza	Dance Studio	Ice Cream Shop	Music Venue	Cocktail Bar	Restaurant
11	Vila Romana	3205.000000	650.000000	265.000000	2.500000	2.500000	139.000000	2	Bar	Pizza Place	Bakery	Gym / Fitness Center	Brazilian Restaurant	Ice Cream Shop	Italian Restaurant	Restaurant	Candy Store	Snack Place
12	Vila Anglo Brasileira	3191.451613	1057.064516	269.655172	2.709677	2.225806	102.064516	31	Brazilian Restaurant	Farmers Market	Burger Joint	Pizza Place	Bar	Dessert Shop	Bakery	Gym	Gym / Fitness Center	Restaurant
14	Jardim Paineiras	3600.000000	2034.000000	96.500000	3.000000	2.000000	150.500000	2	Japanese Restaurant	Café	Fast Food Restaurant	Burger Joint	Clothing Store	BBQ Joint	Sushi Restaurant	Shoe Store	Juice Bar	Brazilian Restaurant
15	Sumaré	3140.000000	1120.000000	266.571429	2.371429	2.142857	91.314286	35	Plaza	Hostel	Pizza Place	Dessert Shop	Burger Joint	Bakery	Music Venue	Bar	Gym	Gym / Fitness Center
16	Jardim Vera Cruz	2250.000000	420.000000	85.000000	2.500000	1.500000	82.500000	2	Food Truck	Grocery Store	Pizza Place	Gym / Fitness Center	Bakery	Gym	Brazilian Restaurant	Café	Pet Store	Gay Bar
17	Jardim Paulistano	3500.000000	1900.000000	300.000000	3.000000	3.000000	93.000000	1	Coffee Shop	Italian Restaurant	Middle Eastern Restaurant	Café	Pizza Place	Bakery	Pet Store	Jewelry Store	Spa	Restaurant
18	Bela Vista	3066.666667	973.333333	307.666667	3.000000	2.000000	85.000000	3	Theater	Bakery	Japanese Restaurant	Bar	Italian Restaurant	Cosmetics Shop	Pizza Place	Brazilian Restaurant	Café	Coffee Shop
19	Liberdade	3033.333333	879.000000	140.000000	2.000000	1.000000	60.000000	3	Japanese Restaurant	Bakery	Cosmetics Shop	Sake Bar	Chinese Restaurant	Bookstore	Gift Shop	Theater	Grocery Store	Asian Restaurant
20	Morada do Sol	3300.000000	880.000000	190.000000	3.000000	2.000000	70.000000	1	Italian Restaurant	Café	Restaurant	Bar	Dessert Shop	Art Gallery	Clothing Store	Pizza Place	Ice Cream Shop	Mineiro Restaurant
21	Pacaembu	3100.000000	1135.000000	334.000000	2.500000	1.500000	100.000000	2	Dessert Shop	Pizza Place	Bakery	Ice Cream Shop	Athletics & Sports	Pet Store	Spa	Plaza	Burger Joint	Park
22	Barra Funda	3272.600000	462.600000	107.333333	2.200000	1.600000	88.400000	5	Brazilian Restaurant	Pizza Place	Restaurant	Café	Dessert Shop	Italian Restaurant	Supermarket	Beer Bar	Farmers Market	Food Truck
23	Itaim Bibi	5000.000000	1400.000000	700.000000	3.000000	4.000000	156.000000	1	Italian Restaurant	Bar	Japanese Restaurant	Restaurant	Hotel	French Restaurant	Cycle Studio	Ice Cream Shop	Burger Joint	Brazilian Restaurant
24	Brás	4000.000000	1700.000000	100.000000	3.000000	1.000000	168.000000	1	Brazilian Restaurant	Clothing Store	Sandwich Place	Italian Restaurant	Food & Drink Shop	Market	Restaurant	Deli / Bodega	Hotel	Health Food Store
Cluster 2
neighborhood_merged.loc[neighborhood_merged['Cluster Labels'] == 1, neighborhood_merged.columns[[0] + list(range(4, neighborhood_merged.shape[1]))]]
Neighborhood	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
0	Perdizes	44128.567201	1335.612422	410.262484	2.65598	2.366215	105.293465	811	Dessert Shop	Pet Store	Pharmacy	Bar	Spa	Burger Joint	Martial Arts School	Athletics & Sports	Pizza Place	Ice Cream Shop
7	Água Branca	39845.812500	721.843750	571.172414	2.12500	1.687500	70.812500	32	Brazilian Restaurant	Bar	Soccer Field	Pizza Place	Convenience Store	Furniture / Home Store	Churrascaria	Gym	Recording Studio	Restaurant
Cluster 3
neighborhood_merged.loc[neighborhood_merged['Cluster Labels'] == 2, neighborhood_merged.columns[[0] + list(range(4, neighborhood_merged.shape[1]))]]
Neighborhood	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
9	Consolação	78372.735849	2149.943396	695.442308	2.90566	2.622642	157.754717	53	Brazilian Restaurant	Gym / Fitness Center	Coffee Shop	Theater	Bakery	Gym	Movie Theater	Bar	Burger Joint	Dance Studio
Cluster 4
neighborhood_merged.loc[neighborhood_merged['Cluster Labels'] == 3, neighborhood_merged.columns[[0] + list(range(4, neighborhood_merged.shape[1]))]]
Neighborhood	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
1	Higienópolis	30550.163498	3905.482890	400.652174	2.631179	2.334601	121.076046	263	Bakery	Athletics & Sports	Coffee Shop	Italian Restaurant	Pizza Place	Middle Eastern Restaurant	Plaza	Dessert Shop	Gym / Fitness Center	Spa
2	Vila Madalena	20661.958904	1469.150685	232.496403	2.260274	1.917808	79.260274	146	Bar	Hostel	Brazilian Restaurant	Pizza Place	Burger Joint	Plaza	Bistro	Restaurant	Ice Cream Shop	Art Gallery
4	Pinheiros	14737.455165	1025.800683	328.864571	2.293984	1.849035	84.118048	881	Art Gallery	Pet Store	Italian Restaurant	Bookstore	Ice Cream Shop	Burger Joint	Café	Bar	Clothing Store	Cocktail Bar
6	Santa Cecília	26533.491329	1693.028902	726.437870	2.924855	2.520231	139.231214	173	Pizza Place	Gym / Fitness Center	Italian Restaurant	Café	Brazilian Restaurant	Beer Bar	Coffee Shop	Art Gallery	Farmers Market	Fruit & Vegetable Store
13	Vila Buarque	27897.692308	695.000000	275.360000	2.115385	1.346154	70.269231	26	Coffee Shop	Brazilian Restaurant	Pizza Place	Bakery	Bar	Italian Restaurant	Hotel	Ice Cream Shop	Middle Eastern Restaurant	Gay Bar
Cluster 5
neighborhood_merged.loc[neighborhood_merged['Cluster Labels'] == 4, neighborhood_merged.columns[[0] + list(range(4, neighborhood_merged.shape[1]))]]
Neighborhood	rent_value	condominium_value	property_tax	bedrooms	bathrooms	private_area	size	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
3	Vila Pompéia	58388.938525	963.443983	529.966387	2.557377	2.139344	81.106557	244	Bar	Italian Restaurant	Pizza Place	Brazilian Restaurant	Ice Cream Shop	Burger Joint	Farmers Market	Coffee Shop	Yoga Studio	Music Venue
Now that we know more about our neighborhoods, let's calculate the nearest specific venue (gyms and markets) to each apartment in our dataset.
First let's merge latitude and longitude in a single row to use the haversine library.

gym_venues = get_location(gym_venues, name_latitude='Venue Latitude', name_longitude='Venue Longitude')
market_venues = get_location(market_venues, name_latitude='Venue Latitude', name_longitude='Venue Longitude')

gym_venues.reset_index(inplace=True)
market_venues.reset_index(inplace=True)
gym_venues.head()
index	Neighborhood	Neighborhood Latitude	Neighborhood Longitude	Venue	Venue Latitude	Venue Longitude	Venue Category	location
0	21	Perdizes	-23.54057	-46.67236	Crossfit Mansion	-23.544026	-46.676688	Gymnastics Gym	[-23.54402617144983, -46.67668784457914]
1	29	Perdizes	-23.54057	-46.67236	Sumaré Sports	-23.541187	-46.679410	Gym	[-23.541187480367938, -46.679409518971006]
2	46	Perdizes	-23.54057	-46.67236	Full Time Academia	-23.535241	-46.670923	Gym	[-23.535241064179097, -46.670922975189974]
3	58	Perdizes	-23.54057	-46.67236	Smart Fit	-23.534156	-46.676010	Gymnastics Gym	[-23.534156461289758, -46.67601027022038]
4	78	Perdizes	-23.54057	-46.67236	Academia Gaviões 24h	-23.539215	-46.680545	Gym / Fitness Center	[-23.539214905009583, -46.68054516901303]
market_venues.head()
index	Neighborhood	Neighborhood Latitude	Neighborhood Longitude	Venue	Venue Latitude	Venue Longitude	Venue Category	location
0	97	Perdizes	-23.54057	-46.67236	Feira Livre	-23.544832	-46.665371	Farmers Market	[-23.544832074201565, -46.665371004879326]
1	188	Higienópolis	-23.54523	-46.65975	Feira Livre	-23.544832	-46.665371	Farmers Market	[-23.544832074201565, -46.665371004879326]
2	195	Higienópolis	-23.54523	-46.65975	Pacífico Pescados	-23.553070	-46.658758	Fish Market	[-23.55307001166797, -46.65875782502824]
3	224	Vila Madalena	-23.55119	-46.69711	Natural da Terra	-23.546568	-46.696114	Farmers Market	[-23.546567613814275, -46.696114061466275]
4	318	Vila Pompéia	-23.53189	-46.68627	Feira Livre	-23.531079	-46.683623	Farmers Market	[-23.531079203669037, -46.68362306812686]
Now let's iterate through every apartment, calculate the distance between every venue and get the nearest distance of it.

# let's create a copy of our original dataset to not modify it
dataset_distance_every_venue = dataset.copy()
for i, apartment in dataset_distance_every_venue.iterrows():
    nearest_gym = []
    nearest_market = []
    for j, gym in gym_venues.iterrows():
        nearest_gym.append((gym['index'], haversine(apartment['location'], gym['location'])))
    for k, market in market_venues.iterrows():
        nearest_market.append((market['index'], haversine(apartment['location'], market['location'])))
    
    nearest_gym_distance = min(map(lambda x: x[1], nearest_gym))
    nearest_market_distance = min(map(lambda x: x[1], nearest_market))

    nearest_gym_element = [i for i in nearest_gym if i[1] == nearest_gym_distance][0]
    nearest_market_element = [i for i in nearest_market if i[1] == nearest_market_distance][0]
    
    dataset_distance_every_venue.loc[i, 'index_nearest_gym'] = nearest_gym_element[0]
    dataset_distance_every_venue.loc[i, 'distance_nearest_gym'] = nearest_gym_element[1]
    dataset_distance_every_venue.loc[i, 'index_nearest_market'] = nearest_market_element[0]
    dataset_distance_every_venue.loc[i, 'distance_nearest_market'] = nearest_market_element[1]
Let's merge our dataset with the gym and market venue datasets to get all the information about the nearest venues.

dataset_distance_every_venue = dataset_distance_every_venue.merge(gym_venues[['index', 'Venue', 'Venue Latitude', 'Venue Longitude']], left_on='index_nearest_gym', right_on='index', suffixes=('_original', '_gym')).sort_values(by='code')
dataset_distance_every_venue.drop(columns = ['index_nearest_gym'], inplace=True)
dataset_distance_every_venue = dataset_distance_every_venue.merge(market_venues[['index', 'Venue', 'Venue Latitude', 'Venue Longitude']], left_on='index_nearest_market', right_on='index', suffixes=('_gym', '_market')).sort_values(by='code')
dataset_distance_every_venue.drop(columns = ['index_nearest_market'], inplace=True)
dataset_distance_every_venue.head()
code	neighborhood	rent_value	condominium_value	property_tax	bedrooms	suites	bathrooms	parking_spaces	private_area	...	distance_nearest_gym	distance_nearest_market	index_gym	Venue_gym	Venue Latitude_gym	Venue Longitude_gym	index_market	Venue_market	Venue Latitude_market	Venue Longitude_market
0	0	Perdizes	2600	1000.0	NaN	2	-	2	1	98	...	0.434775	0.714569	83	Viva Leve Pilates e Fisioterapia	-23.535275	-46.668311	2134	Feira de Produtos Orgânicos da AAO	-23.530314	-46.671194
221	1	Higienópolis	2111	689.0	NaN	2	-	1	-	62	...	0.382325	1.203415	1305	Marquesport	-23.546495	-46.648977	602	Natural da Terra	-23.534688	-46.657594
450	3	Vila Madalena	1900	411.0	85.0	2	-	1	1	48	...	0.534190	0.813728	230	Academia Aquasport	-23.553228	-46.690845	1551	Feira Livre	-23.549318	-46.690142
634	4	Higienópolis	3000	900.0	380.0	3	-	2	1	127	...	0.571180	0.715360	1305	Marquesport	-23.546495	-46.648977	195	Pacífico Pescados	-23.553070	-46.658758
821	5	Vila Pompéia	2150	750.0	122.0	2	-	2	1	56	...	0.400494	0.321486	343	TEAM NOGUEIRA	-23.527387	-46.689057	318	Feira Livre	-23.531079	-46.683623
5 rows × 26 columns

Let's see the shape of the dataset

dataset_distance_every_venue.shape
(2906, 26)
Finally, let's filter our dataset to fill our pre requisites

the monthly value of the rent cannot exceed R$ 1.700,00
it has to be near at least 1km from a gym and a market
the apartment must have at least 2 bedrooms
final_dataset = dataset_distance_every_venue.loc[(dataset_distance_every_venue['distance_nearest_gym'] < min_dist_gym) & (dataset_distance_every_venue['distance_nearest_market'] < min_dist_mkt) & (dataset_distance_every_venue['rent_value'].astype(int) < max_rent_value) & (dataset_distance_every_venue['bedrooms'] >= min_bedrooms)]
final_dataset.head()
code	neighborhood	rent_value	condominium_value	property_tax	bedrooms	suites	bathrooms	parking_spaces	private_area	...	distance_nearest_gym	distance_nearest_market	index_gym	Venue_gym	Venue Latitude_gym	Venue Longitude_gym	index_market	Venue_market	Venue Latitude_market	Venue Longitude_market
1693	326	Pinheiros	1500	1800.0	300.0	2	1	2	1	45	...	0.233313	0.990337	1675	Bodytech	-23.572642	-46.696177	1681	Mercado Municipal de Pinheiros	-23.565697	-46.692589
1314	1459	Pinheiros	1500	505.0	0.0	2	-	1	1	40	...	0.467882	0.658186	433	Concept Academia	-23.564832	-46.682705	1927	Feira Rosenbaum	-23.560642	-46.675947
1137	2819	Pinheiros	1650	516.0	50.0	2	0	1	0	55	...	0.528848	0.277739	432	Academia Esporte 120	-23.565341	-46.684860	440	Feira Livre da Vila Madalena	-23.560219	-46.689883
99	3835	Perdizes	1500	400.0	230.0	2	-	1	-	70	...	0.256925	0.713609	46	Full Time Academia	-23.535241	-46.670923	2134	Feira de Produtos Orgânicos da AAO	-23.530314	-46.671194
1168	4614	Pinheiros	1500	550.0	NaN	2	0	1	1	41	...	0.403543	0.630450	433	Concept Academia	-23.564832	-46.682705	440	Feira Livre da Vila Madalena	-23.560219	-46.689883
5 rows × 26 columns

print(f'We have {final_dataset.shape[0]} apartments that fit our profile')
We have 17 apartments that fit our profile
Let's then create a map with the apartments that fit our profile and the nearest venues of each. Let's also add a circle radius every 1km.

final_map = folium.Map(location=[work_location[0], work_location[1]], zoom_start=14)

for lat, lng, label, distance_workspace, distance_gym, distance_mkt, gym_lat, gym_lng, mkt_lat, mkt_lng, venue_gym, venue_mkt, rent_value in zip(final_dataset['latitude'], final_dataset['longitude'], final_dataset['code'], final_dataset['distance_from_workplace'], final_dataset['distance_nearest_gym'], final_dataset['distance_nearest_market'], final_dataset['Venue Latitude_gym'], final_dataset['Venue Longitude_gym'], final_dataset['Venue Latitude_market'], final_dataset['Venue Longitude_market'], final_dataset['Venue_gym'], final_dataset['Venue_market'], final_dataset['rent_value']):
    folium.Marker(
        [lat, lng],
        radius=5,
        popup=f'code={label}\ndistance_for_workspace={round(distance_workspace, 2)}km\ndistance_gym={round(distance_gym, 2)}km\ndistance_mkt={round(distance_mkt,2)}km\nrent_value=R${round(rent_value,2)}',
        icon=folium.Icon(icon='home', color="blue"),
    ).add_to(final_map)

    folium.CircleMarker(
        [gym_lat, gym_lng],
        radius=10,
        popup=venue_gym,
        fill=True,
        color='red',
        fill_color='red',
        fill_opacity=0.6
    ).add_to(final_map)

    folium.CircleMarker(
        [mkt_lat, mkt_lng],
        radius=10,
        popup=venue_mkt,
        fill=True,
        color='green',
        fill_color='green',
        fill_opacity=0.6
    ).add_to(final_map)

folium.Marker(
    [work_location[0], work_location[1]],
    radius=5,
    tooltip='Workplace',
    icon=folium.Icon(icon='briefcase', color="green")
).add_to(final_map)

# add the 5km radius
for radius in range(1000, 6000, 1000):
    folium.Circle([work_location[0], work_location[1]],
                        radius=radius,
                        color='white'
                    ).add_to(final_map)

final_map
Make this Notebook Trusted to load map: File -> Trust Notebook
Finally, let's merge our potential choices with our neighborhood data to merge all information about the propertie and it's neighborhood.

choice_dataset = final_dataset.drop(columns=['latitude', 'longitude', 'location', 'index_gym', 'Venue Latitude_gym', 'Venue Longitude_gym', 'index_market', 'Venue Latitude_market', 'Venue Longitude_market'])
choice_dataset = choice_dataset.merge(neighborhood_social_economic.set_index('neighborhood'), on='neighborhood', suffixes=('_apartment', '_avg_neighborhood'))
Finally, we have a dataset for all of our potential choices with the data from the apartment and the neighborhood.

Now we are ready for choose our new home.

pd.set_option('display.max_columns', choice_dataset.shape[1])
choice_dataset
code	neighborhood	rent_value_apartment	condominium_value_apartment	property_tax_apartment	bedrooms_apartment	suites	bathrooms_apartment	parking_spaces	private_area_apartment	furnished	amenities	distance_from_workplace	distance_nearest_gym	distance_nearest_market	Venue_gym	Venue_market	Cluster Labels	rent_value_avg_neighborhood	condominium_value_avg_neighborhood	property_tax_avg_neighborhood	bedrooms_avg_neighborhood	bathrooms_avg_neighborhood	private_area_avg_neighborhood	size	1st Most Common Venue	2nd Most Common Venue	3rd Most Common Venue	4th Most Common Venue	5th Most Common Venue	6th Most Common Venue	7th Most Common Venue	8th Most Common Venue	9th Most Common Venue	10th Most Common Venue
0	326	Pinheiros	1500	1800.0	300.0	2	1	2	1	45	True	['FURNISHED', 'POOL', 'GYM', 'BARBECUE_GRILL',...	2.245463	0.233313	0.990337	Bodytech	Mercado Municipal de Pinheiros	3	14737.455165	1025.800683	328.864571	2.293984	1.849035	84.118048	881	Art Gallery	Pet Store	Italian Restaurant	Bookstore	Ice Cream Shop	Burger Joint	Café	Bar	Clothing Store	Cocktail Bar
1	1459	Pinheiros	1500	505.0	0.0	2	-	1	1	40	False	['ELEVATOR', 'PETS_ALLOWED']	1.087836	0.467882	0.658186	Concept Academia	Feira Rosenbaum	3	14737.455165	1025.800683	328.864571	2.293984	1.849035	84.118048	881	Art Gallery	Pet Store	Italian Restaurant	Bookstore	Ice Cream Shop	Burger Joint	Café	Bar	Clothing Store	Cocktail Bar
2	2819	Pinheiros	1650	516.0	50.0	2	0	1	0	55	False	['ELEVATOR']	0.847613	0.528848	0.277739	Academia Esporte 120	Feira Livre da Vila Madalena	3	14737.455165	1025.800683	328.864571	2.293984	1.849035	84.118048	881	Art Gallery	Pet Store	Italian Restaurant	Bookstore	Ice Cream Shop	Burger Joint	Café	Bar	Clothing Store	Cocktail Bar
3	4614	Pinheiros	1500	550.0	NaN	2	0	1	1	41	False	['GARAGE']	1.046584	0.403543	0.630450	Concept Academia	Feira Livre da Vila Madalena	3	14737.455165	1025.800683	328.864571	2.293984	1.849035	84.118048	881	Art Gallery	Pet Store	Italian Restaurant	Bookstore	Ice Cream Shop	Burger Joint	Café	Bar	Clothing Store	Cocktail Bar
4	3835	Perdizes	1500	400.0	230.0	2	-	1	-	70	False	['SERVICE_AREA', 'PETS_ALLOWED']	2.626068	0.256925	0.713609	Full Time Academia	Feira de Produtos Orgânicos da AAO	1	44128.567201	1335.612422	410.262484	2.655980	2.366215	105.293465	811	Dessert Shop	Pet Store	Pharmacy	Bar	Spa	Burger Joint	Martial Arts School	Athletics & Sports	Pizza Place	Ice Cream Shop
5	4773	Perdizes	1500	1315.0	282.0	2	0	1	1	64	False	['POOL', 'ELEVATOR', 'SERVICE_AREA', 'PLAYGROU...	2.927133	0.245457	0.371183	Full Time Academia	Feira de Produtos Orgânicos da AAO	1	44128.567201	1335.612422	410.262484	2.655980	2.366215	105.293465	811	Dessert Shop	Pet Store	Pharmacy	Bar	Spa	Burger Joint	Martial Arts School	Athletics & Sports	Pizza Place	Ice Cream Shop
6	5883	Perdizes	1600	524.0	1630.0	2	0	1	0	80	False	[]	2.436784	0.297822	0.786187	Smart Fit	Defrost	1	44128.567201	1335.612422	410.262484	2.655980	2.366215	105.293465	811	Dessert Shop	Pet Store	Pharmacy	Bar	Spa	Burger Joint	Martial Arts School	Athletics & Sports	Pizza Place	Ice Cream Shop
7	5949	Perdizes	1500	970.0	1031.0	2	0	2	0	80	False	['ELEVATOR', 'GATED_COMMUNITY', 'GARDEN']	1.951137	0.218856	0.644863	Smart Fit	Defrost	1	44128.567201	1335.612422	410.262484	2.655980	2.366215	105.293465	811	Dessert Shop	Pet Store	Pharmacy	Bar	Spa	Burger Joint	Martial Arts School	Athletics & Sports	Pizza Place	Ice Cream Shop
8	6268	Perdizes	1680	650.0	NaN	2	0	1	0	60	False	['ELEVATOR', 'PARTY_HALL', 'BUILTIN_WARDROBE',...	3.067646	0.110805	0.731746	Viva Leve Pilates e Fisioterapia	Feira de Produtos Orgânicos da AAO	1	44128.567201	1335.612422	410.262484	2.655980	2.366215	105.293465	811	Dessert Shop	Pet Store	Pharmacy	Bar	Spa	Burger Joint	Martial Arts School	Athletics & Sports	Pizza Place	Ice Cream Shop
9	4706	Vila Pompéia	1500	1290.0	181.0	2	0	2	1	78	False	['ELEVATOR', 'PARTY_HALL', 'BUILTIN_WARDROBE']	2.750144	0.180373	0.429239	CPN Academia	Feira Livre	4	58388.938525	963.443983	529.966387	2.557377	2.139344	81.106557	244	Bar	Italian Restaurant	Pizza Place	Brazilian Restaurant	Ice Cream Shop	Burger Joint	Farmers Market	Coffee Shop	Yoga Studio	Music Venue
10	4804	Água Branca	1699	460.0	NaN	2	1	1	0	40	False	['ELEVATOR', 'PARTY_HALL']	3.222871	0.356279	0.512169	Casa de Pedra	Feira de Produtos Orgânicos da AAO	1	39845.812500	721.843750	571.172414	2.125000	1.687500	70.812500	32	Brazilian Restaurant	Bar	Soccer Field	Pizza Place	Convenience Store	Furniture / Home Store	Churrascaria	Gym	Recording Studio	Restaurant
11	4935	Água Branca	1150	960.0	NaN	2	0	1	0	50	False	['ELEVATOR', 'BUILTIN_WARDROBE', 'KITCHEN_CABI...	3.212717	0.370528	0.461818	Casa de Pedra	Feira de Produtos Orgânicos da AAO	1	39845.812500	721.843750	571.172414	2.125000	1.687500	70.812500	32	Brazilian Restaurant	Bar	Soccer Field	Pizza Place	Convenience Store	Furniture / Home Store	Churrascaria	Gym	Recording Studio	Restaurant
12	7224	Água Branca	1600	960.0	0.0	2	0	1	1	48	False	[]	3.212717	0.370528	0.461818	Casa de Pedra	Feira de Produtos Orgânicos da AAO	1	39845.812500	721.843750	571.172414	2.125000	1.687500	70.812500	32	Brazilian Restaurant	Bar	Soccer Field	Pizza Place	Convenience Store	Furniture / Home Store	Churrascaria	Gym	Recording Studio	Restaurant
13	5315	Vila Madalena	1275	2359.0	511.0	3	1	2	2	96	True	['FURNISHED', 'POOL', 'BARBECUE_GRILL', 'ELEVA...	1.827664	0.171577	0.436439	Ecofit	Minuto Pão De Açúcar	3	20661.958904	1469.150685	232.496403	2.260274	1.917808	79.260274	146	Bar	Hostel	Brazilian Restaurant	Pizza Place	Burger Joint	Plaza	Bistro	Restaurant	Ice Cream Shop	Art Gallery
14	5396	Vila Madalena	1600	450.0	0.0	2	0	2	0	69	False	[]	1.375651	0.498940	0.244829	Tracer Parkour	Minuto Pão De Açúcar	3	20661.958904	1469.150685	232.496403	2.260274	1.917808	79.260274	146	Bar	Hostel	Brazilian Restaurant	Pizza Place	Burger Joint	Plaza	Bistro	Restaurant	Ice Cream Shop	Art Gallery
15	6670	Higienópolis	1600	1103.0	220.0	2	0	2	1	50	False	['GYM', 'GARAGE', 'ELEVATOR', 'KITCHEN']	3.796113	0.498101	0.822581	Marquesport	Pacífico Pescados	3	30550.163498	3905.482890	400.652174	2.631179	2.334601	121.076046	263	Bakery	Athletics & Sports	Coffee Shop	Italian Restaurant	Pizza Place	Middle Eastern Restaurant	Plaza	Dessert Shop	Gym / Fitness Center	Spa
16	6938	Higienópolis	1572	2700.0	750.0	2	2	2	2	70	True	['FURNISHED', 'POOL', 'AIR_CONDITIONING', 'ELE...	2.993272	0.625510	0.254718	Bio Ritmo	Feira Livre	3	30550.163498	3905.482890	400.652174	2.631179	2.334601	121.076046	263	Bakery	Athletics & Sports	Coffee Shop	Italian Restaurant	Pizza Place	Middle Eastern Restaurant	Plaza	Dessert Shop	Gym / Fitness Center	Spa
Results and Discussion
Our analysis show that although there are a great number of apartments (~3.000) in our original dataset, just a few (17) fits our profile. Our first filter with the aim of get only apartments in a 5km radius for our workplace did not filter a lot of properties (~100) and that's due we already had filtered the neighborhoods to scrape the data.

The heatmpap show that there are clearly concentration of the apartments in the neighboorhods that we early filtered, resulting in a little bias in our results. To next projects we could consider get all neighborhoods in a 5km radius from our starting point.

After we grouped the economic and social data of the neighborhoods, we clustered the neighborhoods into 5 groups. Apparently, the neighborhoods are very similar to each other, resulting in a high concentration in one of the groups (~64% of the neighborhoods).

To find our potencial apartments, we applied the final filters regarded as number of bedrooms, minimum distance from specific venues and maximum monthly rent value. The filter were very effective, filtering approximately 99,4% of the results.

To generate our final dataset, we grouped every apartment data with the information of the nearest venues of interest and the economical and social data of the neighborhood. This provided us with a broad view of the characteristics of each apartment and its surroundings.

Conclusion
The objective of this project was to find the best apartment for rent in São Paulo that fits our profile. By putting together data of apartments in near neighborhoods and extracting the social and economic data from these neighborhoods, we have generated a concentrated dataset with the most promising apartments to live in.

The final decision should always be taken carefully after looking at each apartment individually. The project was helpful to filter the great number of apartments and grouping together the data of every potential apartment and the social and economical data of it respective neighborhood. Data of quality of life, criminality, urban mobility etc was not taken into consideration and could be interesting indicators to look after in next projects to improve the analysis.
