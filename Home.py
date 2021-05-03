#Importing libraries
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import matplotlib as plt
import seaborn as sn

# Setting option to pd to show all columns at any given time without hiding
pd.set_option('display.max_columns',None)


#Loading the dataset
movies = pd.read_csv('/Users/thejakamahaulpatha/PycharmProjects/MovieRecommendation/Datasets/movies.csv')
ratings = pd.read_csv('/Users/thejakamahaulpatha/PycharmProjects/MovieRecommendation/Datasets/ratings.csv')

#Descriptive Analysis of data
print(movies.shape,ratings.shape)
print(movies.describe())
print(ratings.describe())

#Pivot the rating dataframe to get a better knowledge
final_data = ratings.pivot(index='movieId',columns='userId',values='rating')
# print((final_data))

#Let's replace the NAN with 0

final_data.fillna(0,inplace=True)
print(final_data.head())

#Remove Noise from the Data
#To qualify a movie, a minimum of 10 users should have voted a movie.
#To qualify a user, a minimum of 50 movies should have voted by the user.

no_users_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = movies.groupby('userId')['rating'].agg('count')

