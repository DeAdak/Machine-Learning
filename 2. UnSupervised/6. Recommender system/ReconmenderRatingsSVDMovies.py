# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:26:28 2022

@author: R
"""

import pandas as pd
import numpy as np
import surprise


movie=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/ml-100k/u.data",sep='\t',
                  names=['user id','item id','rating','timestamp'])
movie=movie.drop('timestamp',axis=1)
min_rating=movie['rating'].min()
max_rating=movie['rating'].max()
print(f"rating range is between {min_rating} and {max_rating}")
reader= surprise.Reader(rating_scale=(min_rating,max_rating))
data=surprise.Dataset.load_from_df(movie, reader)
similarity_options = {'name':'cosine','user_based':True}
algo = surprise.SVD(random_state=(2022))
output = algo.fit(data.build_full_trainset())

iids=movie['item id'].unique()
iids100 = movie.loc[movie['user id']==100,'item id']
# iids that are not purchased by uid 100
iids_to_predict = np.setdiff1d(iids,iids100)

testset=[[100,iid,0] for iid in iids_to_predict]
# make a complete set of non purchased items for uid 100

predictions = algo.test(testset,verbose=True)
pred_ratings=np.array([pred.est for pred in predictions])
# returns index no of highest predicted rating
i_max = pred_ratings.argmax()
predictions[i_max]
iids_to_predict[i_max]

import heapq
i_sorted_10 = heapq.nlargest(10,range(len(pred_ratings)),pred_ratings.take)
# returns indices no of 10 highest predicted rating
top_10_items = iids_to_predict[i_sorted_10]

############ Tuning ############

from surprise.model_selection import GridSearchCV
param_grid = {'n_epochs':np.arange(5,50,10),'lr_all':np.linspace(.001,1,5),'reg_all':np.linspace(0.01,0.8,5)}

from surprise.model_selection.split import KFold
kfold = KFold(n_splits=5, random_state=2022, shuffle=True)
gs = GridSearchCV(surprise.SVD, param_grid, 
                  measures=['rmse', 'mae'], cv=kfold,joblib_verbose=2,n_jobs=-1)

gs.fit(data)
# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])