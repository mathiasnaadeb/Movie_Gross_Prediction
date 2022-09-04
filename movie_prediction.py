
######### Movie Data collection #####################
## The data collected is text data from website imdb.com

from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from pandas import Series, DataFrame




html_text = requests.get('https://www.imdb.com/search/title/?title_type=feature&num_votes=100000,&languages=en&sort=user_rating,desc&count=250').text
soup = BeautifulSoup(html_text, 'lxml')


## Create container that contains all required data
container_all = soup.find_all('div',class_="lister-item mode-advanced" )


#Title of the movie
    Title = container_all[0].h3.a.string
    Title
    
# Runtime
    Runtime = container_all[0].find('span', class_='runtime').string.split(' ')[0]
    Runtime
    
# Genre of the movie
    Genre = container_all[0].find('span',class_='genre').string[1:]
    Genre
    
# Movie Director
    Director = container_all[0].find('p', class_='').find_all('a')[0].string
    Director
    
#Movie Cast
    Casts = container_all[0].find('p',class_='').getText().rpartition(':')[2].split('\n')[1:][:-1]
    Casts
    
# Gross of the movie
    Gross_container = container_all[0].find_all('span', attrs = {'name':'nv'})
    Gross= Gross_container[1].text[1:][:-1] if len(Gross_container) > 1 else 'NA'
    Gross

# Rating of the movie
    imdb_rating=float(container_all[0].strong.text)
    imdb_rating 

## Year of release
    Year = container_all[0].find('span',class_="lister-item-year text-muted unbold").text[1:5]
    Year

# Budget of the movie
    ids = container_all[0].find('a').get('href')
    new_url = 'https://www.imdb.com'+ ids + '?ref_=adv_li_tt' # the url where the budget is found
    new_html = requests.get(new_url).text
    new_soup = BeautifulSoup(new_html, 'lxml')
    new_container_all = new_soup.find_all('span', class_="ipc-metadata-list-item__list-content-item" )
    Budget_1 = new_container_all[2].text
    Budget = Budget_1[1:].rsplit(" ")[0]
    Budget
    



## Now lets use a for loop to scrape all titles, runtime, genres, directors, actors, gross and ratings

## Create an empty list
Movie_Title = []
Movie_Runtime = []
Movie_Genre = []
Movie_imdb_rating = []
Movie_Gross = []
Movie_Director = []
Movie_Casts = []
Movie_year = []
Movie_Budget = []

for my_container in container_all:

#Title
    Title = my_container.h3.a.string
    Movie_Title.append(Title)
# Runtime
    Runtime = my_container.find('span', class_='runtime').string.split(' ')[0]
    Movie_Runtime.append(Runtime)
# Genre
    Genre = my_container.find('span',class_='genre').string[1:]
    Movie_Genre.append(Genre)
# Movie Director
    Director = my_container.find('p', class_='').find_all('a')[0].string
    Movie_Director.append(Director)
#Movie Cast
    Casts = my_container.find('p',class_='').getText().rpartition(':')[2].split('\n')[1:][:-1]
    Movie_Casts.append(Casts)
# Gross
    Gross_container = my_container.find_all('span', attrs = {'name':'nv'})
    Gross= Gross_container[1].text[1:][:-1] if len(Gross_container) > 1 else 'NaN'
    Movie_Gross.append(Gross)
# Rating
    imdb_rating=float(my_container.strong.text)
    Movie_imdb_rating.append(imdb_rating)

# Movie released Year
    Year = my_container.find('span',class_="lister-item-year text-muted unbold").text[1:5]
    Movie_year.append(Year)



### Scraping Movie Budget
movie_id = []  # Empty list to store the movie ids
# For loop to get all ids
for my_container in container_all:
    ids = my_container.find('a').get('href')
    movie_id.append(ids) # This contains all the movie ids to the budget page

## for loop to scrape movie budget
#movie_id = container_all[0].find('a').get('href') 
for i in range(len(movie_id)) :
    new_url = 'https://www.imdb.com'+ movie_id[i] + '?ref_=adv_li_tt' # budget
    new_html = requests.get(new_url).text
    new_soup = BeautifulSoup(new_html, 'lxml')
    new_container_all = new_soup.find_all('span', class_="ipc-metadata-list-item__list-content-item" )
    Budget_1 = new_container_all[2].text
    Budget = Budget_1[1:].rsplit(" ")[0]
    Movie_Budget.append(Budget)


# Renaming the columns
movie = {'title':Movie_Title,
            'runtime':Movie_Runtime, 
            'genre':Movie_Genre, 
            'rating':Movie_imdb_rating, 
            'gross':Movie_Gross, 
            'director':Movie_Director, 
            'Top_actors':Movie_Casts,
            'year' : Movie_year,
            'budget': Movie_Budget} 
             
## Puting data to a dataframe format                                          
Movie_data = pd.DataFrame(movie)


##### Cleaning data
Movie_data.head()

## Checking data info
Movie_data.info()

# Converting Genre to string
Movie_data['genre'] = Movie_data['genre'].apply(str)

## Remove the square brackets in Top_actors
Movie_data['Top_actors']= Movie_data['Top_actors'].astype(str).replace({"\[":"", "\]":""}, regex=True)

## Remove unnecessary quotation marks and commas from Top_actors
Movie_data['Top_actors'] = Movie_data['Top_actors'].apply(lambda x: x.replace("',", ''))
Movie_data['Top_actors'] = Movie_data['Top_actors'].apply(lambda x: x.replace("'", ''))

#converting gross to numeric
Movie_data['gross'] = pd.to_numeric(Movie_data['gross'])
## Rows with NaN
Movie_data['gross'].loc[Movie_data['gross'] == 'NaN']
# Delete columns with NaN
Movie_data.drop(Movie_data.index[Movie_data["gross"] == "NaN"], inplace = True)
# Now convert gross to numeric
Movie_data['gross'] = pd.to_numeric(Movie_data['gross'])


## converting Year to date format
Movie_data['year']= pd.to_datetime(Movie_data['year'])
Movie_data["year"]=pd.to_datetime(Movie_data["year"], errors = "coerce").fillna(0)
Movie_data.drop(Movie_data.index[Movie_data["year"] == 0], inplace = True)


# converting budget variables to float and dividing by 1000000 (million)
Movie_data['budget'] = Movie_data['budget'].apply(lambda x: x.replace(",", ''))
Movie_data['budget']
# Removing wrong variables
Movie_data = Movie_data.drop(labels=[38,101,185])

movie_budget = []
for i in Movie_data['budget'][0:]: 
      new_B = float(i)/1000000     
      movie_budget.append(new_B)

Movie_data['movie_budget']= movie_budget
del Movie_data['budget']


# converting runtime to numeric
Movie_data['runtime'] = pd.to_numeric(Movie_data['runtime'])


# Calculating Profit and adding to the data
Profit = Movie_data['gross'] - Movie_data['movie_budget']
Movie_data['Profit'] = Profit

Movie_data.head()

# correlation between the variables runtime, rating, budget and gross
Corr_Movie_data = Movie_data[['runtime', 'rating', 'movie_budget', 'gross']]
sns.heatmap(Corr_Movie_data.corr(), cmap='YlGnBu', annot=True, linewidths = 0.2)
## From the correlation plot, it implies that the budget correlates the most (0.49) with gross(revenue),
## runtime and ratings both had a correlation of (0.12) with the gross


# plot gross against year
plt.figure(figsize=(15, 5))
plt.scatter(x=Movie_data['year'], y=Movie_data['gross'], color='red', marker='o')
plt.title('Gross trend over the years')
plt.xlabel('Years')
plt.ylabel('Gross')
# Gross or revenue increases with the years



# plot budget against year
plt.figure(figsize=(15, 5))
plt.scatter(x=Movie_data['year'], y=Movie_data['movie_budget'], color='red', marker='o')
plt.title('Budget trend over the years')
plt.xlabel('Years')
plt.ylabel('Budget')
# Budget also increases with the years


# plot Top_actors against Profit
plt.figure(figsize=(15, 5))
plt.scatter(x=Movie_data['Top_actors'], y=Movie_data['Profit'], color='red', marker='o')
plt.title('Actors against profit')
plt.xlabel('Top_actors')
plt.ylabel('Profit')


## Profit against Budgets
plt.figure(figsize=(30, 5))
plt.scatter(x=Movie_data['movie_budget'], y=Movie_data['Profit'], color='red', marker='o')
plt.title('A plot of Budget against profit')
plt.xlabel('Budget')
plt.ylabel('Profit')
# An increase in Budget increases the profit


## Spliting the movie genre and converting to categorical variables
genre_split = Movie_data['genre'].str.split(',').apply(Series, 1).stack()
genre_split.index = genre_split.index.droplevel(-1)
genre_split.name = 'genre'
del Movie_data['genre']
Movie_data = Movie_data.join(genre_split)
Movie_data.head()
Movie_data.info()


###############################################################################

### What kind of films (thriller, adventure, romantic,...) should you
#   create to make sure you make the most money?

# First We group the movies by genre using title as unique identifier 
Movie_genre = (pd.DataFrame(Movie_data.groupby('genre').title.nunique())).sort_values('title', ascending=True)
len(Movie_genre)
#Movie_genre = Movie_data.groupby('genre')['title'].count().sort_values(ascending=True)
##Plotting a bar graph of the movie genres
Movie_genre['title'].plot.barh(title = 'Movies by Genre',color='DarkBlue', figsize=(15, 9))
# The most category of movie genre produced is Drama

# Group data by genre and get mean for each genre and each variable
Movie_genre_mean = Movie_data.groupby(['genre']).mean()
Movie_genre_mean.sort_values('movie_budget', ascending=True, inplace = True )
Movie_genre_mean[['gross', 'movie_budget']].plot.barh(stacked=False, title = 'Budget and Gross by Genre (US$ million)',color=('red','c'), figsize=(15, 10))

Movie_genre_mean.sort_values('Profit', ascending=True, inplace = True )
Movie_genre_mean['Profit'].plot.barh(stacked=False, title = 'Profit by Genre (US$ million)',color='DarkBlue', figsize=(10, 9))
# Fantasy movie had the highest profit


###############################################################################


## Who are the top 3 actors you should hire to make sure you make the most money?

## Spliting the movie Top actors into individuals
Top_actors_split = Movie_data['Top_actors'].str.split(',').apply(Series, 1).stack()
Top_actors_split.index = Top_actors_split.index.droplevel(-1)
Top_actors_split.name = 'Top_actors'
del Movie_data['Top_actors']
Movie_data = Movie_data.join(Top_actors_split)
Movie_data.head()

## Grouping the actors by title and converting to dataframe
Movie_Top_actors = (pd.DataFrame(Movie_data.groupby('Top_actors').title.nunique())).sort_values('title', ascending=True)
len(Movie_Top_actors)
## Plotting a bar graph of the movie genres
Movie_Top_actors['title'].plot.barh(title = 'Movies by Actors',color='DarkBlue', figsize=(12, 100))
# The actor with the most most movie featured in, is Leonardo Di Caprio

## Group data by actors and get mean for all the variables 
Movie_Top_actors_mean = Movie_data.groupby(['Top_actors']).mean()
Movie_Top_actors_mean.sort_values('Profit', ascending=True, inplace = True ) #sort the profits in ascending order
## Plot a graph of profits by actors
Movie_Top_actors_mean['Profit'].plot.barh(stacked=False, title = 'Profit by actors (US$ million)',color='DarkBlue', figsize=(12, 100))
## The top 3 actors that featured in movies that made the highest profit per the data 
## is Jacob Batalon, Tom Holland and Benedict Cumberbach. Therefore one can consider hiring them
## if one wants to make much profit



###############################################################################

## Which film director and film producer should you hire to make sure
## you make the most money?

Movie_director = (pd.DataFrame(Movie_data.groupby('director').title.nunique())).sort_values('title', ascending=True)
len(Movie_director)
##Plotting a bar graph of the movie directors
Movie_director['title'].plot.barh(title = 'Movies by director',color='DarkBlue', figsize=(12, 20))
# The director with the most produced movie by title is Christopher Nolan


# Group data by director and get the mean for each director
Movie_director_mean = Movie_data.groupby(['director']).mean()
## Sort prfit in ascending oder
Movie_director_mean.sort_values('Profit', ascending=True, inplace = True )
Movie_director_mean['Profit'].plot.barh(stacked=False, title = 'Profit by director (US$ million)',color='DarkBlue', figsize=(12, 20))
## Jon Watts is the movie director with the most profit. Therefor to make the most money, one should 
## hire Jon Watts as the movie director


###############################################################################

# How much do you need to spend to produce the movie?

Movie_budget_mean = Movie_data.mean()
Movie_budget_mean.sort_values()
# The mean budget required to produce a movie is 56.707670


## Profit against Budgets
plt.figure(figsize=(15, 5))
plt.scatter(x=Movie_data['movie_budget'], y=Movie_data['Profit'], color='red', marker='o')
plt.title('A plot of Budget against profit')
plt.xlabel('Budget')
plt.ylabel('Profit')
 

####################################################################################

## Building a model

# A simple linear regression to predict the gross
Model_data = Movie_data
Model_data.info()
Model_data.head()


# Converting director to categorical variable
Model_data['director'] = Model_data['director'].astype('category')

# Converting genre to categorical variable
Model_data['genre'] = Model_data['genre'].astype('category')

# Converting Top_actors to categorical variable
Model_data['Top_actors'] = Model_data['Top_actors'].astype('category')

# Converting title to categorical variable
Model_data['title'] = Model_data['title'].astype('string')


# check Data information
Model_data.head()


## Packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.preprocessing import scale




# Dependent variable is Gross and taking the log of it
y = np.log1p(Model_data["gross"])
print(y)

# Taking the log of the budget
Model_data['movie_budget'] = np.log1p(Model_data['movie_budget'])

# Independent variables used are runtime, rating, director, year, genre, Top_actors
X = Model_data[["runtime","movie_budget", "rating", "genre","Top_actors", "director"]]
print(X)

# Treating categrical variables by introducing dummy variables
X = pd.get_dummies(data=X, drop_first=True)
X.head()


# Normalising the variables
X = scale(X)
y = scale(y)

# Spliting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=134)
  

######################################################### 


## RandomForest Regression

from sklearn.ensemble import RandomForestRegressor

# Fit Model
Model_RF = RandomForestRegressor(random_state =0, n_estimators=500, max_depth=10, bootstrap=(100))

# Fit and train model
Model_RF.fit(X_train,y_train)
Model_RF_score = Model_RF.score(X_train, y_train)

# Predict test data using trained model
RF_y_pred = Model_RF.predict(X_test)

mse_RF_pred = mean_squared_error(y_test, RF_y_pred)
mae_RF_pred = mean_absolute_error(y_test, RF_y_pred)
# Metrics
print("MSE: ", mse_RF_pred)
print("RMSE: ", mse_RF_pred*(1/2.0)) 
print("MAE: ", mae_RF_pred)


# MSE:  0.0573220806991717
# RMSE: 0.02866104034958585
# MAE:  0.16350787949274653



####################################################################################################

# K nearest Neighbor
from sklearn.neighbors import KNeighborsRegressor

#  model
model_Knn = KNeighborsRegressor(n_neighbors=3)

# fit and train the model using the training data and training targets
model_Knn.fit(X_train, y_train)

## predict y give X_test
Knn_y_pred = model_Knn.predict(X_test)

# Metrics
mse_Knn_pred = mean_squared_error(y_test, Knn_y_pred)
mae_Knn_pred = mean_absolute_error(y_test, Knn_y_pred)


print("MSE: ", mse_Knn_pred)
print("RMSE: ", mse_Knn_pred*(1/2.0)) 
print("MAE: ", mae_Knn_pred)

# MSE:  0.037443363333499055
# RMSE:  0.018721681666749528
# MAE:  0.053391674498238964


#################################################################################

## Support vector Machines
from sklearn.svm import SVR

# Model
Model_svm = SVR(kernel='rbf')

# Fit and train model
Model_svm.fit(X_train, y_train)

## predict y give X_test
Svm_y_pred = Model_svm.predict(X_test)

# Metrics
mse_Svm_pred = mean_squared_error(y_test, Svm_y_pred)
mae_Svm_pred = mean_absolute_error(y_test, Svm_y_pred)


print("MSE: ", mse_Svm_pred)
print("RMSE: ", mse_Svm_pred*(1/2.0)) 
print("MAE: ", mae_Svm_pred)

# MSE:  0.030174123612520847
# RMSE:  0.015087061806260424
# MAE:  0.11179872909552081

########################################################################################

## Gradient Boosting regression

from sklearn.ensemble import GradientBoostingRegressor

Model_GBR = GradientBoostingRegressor(n_estimators=500, 
            max_depth=5, learning_rate=0.01, min_samples_split=3)

# Fit and train model    
Model_GBR.fit(X_train, y_train)    

## predict y given X_test
GBR_y_pred = Model_GBR.predict(X_test)

# Metrics
mse_GBR_pred = mean_squared_error(y_test, GBR_y_pred)
mae_GBR_pred = mean_absolute_error(y_test, GBR_y_pred)


print("MSE: ", mse_GBR_pred)
print("RMSE: ", mse_GBR_pred*(1/2.0)) 
print("MAE: ", mae_GBR_pred)

# MSE:  0.06358853872958763
# RMSE: 0.03179426936479381
# MAE:  0.21617752402886353







