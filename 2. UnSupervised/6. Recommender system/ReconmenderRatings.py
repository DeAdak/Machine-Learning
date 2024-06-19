# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:49:48 2022

@author: R
"""
import pandas as pd
import numpy as np
import surprise

rating_df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/dayWise/UnSupervised/Day15/filmtrust/ratings.txt"
                   ,sep=" ",names=['uid','iid','rating'])
min_rating=rating_df['rating'].min()
max_rating=rating_df['rating'].max()
print(f"rating range is between {min_rating} and {max_rating}")
scale= surprise.Reader(rating_scale=(min_rating,max_rating))
surprise_data=surprise.Dataset.load_from_df(rating_df, scale)
similarity_options = {'name':'cosine','user_based':True}
algo = surprise.KNNBasic(sim_options=similarity_options)
output = algo.fit(surprise_data.build_full_trainset())
# .fit is going to find the ?, k = 40(default)
pred=algo.predict(uid='53',iid='3')
print(pred.est)

iids=rating_df['iid'].unique()

iids50s = rating_df[rating_df['uid']==50]
iids50 = rating_df.loc[rating_df['uid']==50,'iid']
# iids that are not purchased by uid 50
iids_to_predict = np.setdiff1d(iids,iids50)

testset=[[50,iid,0] for iid in iids_to_predict]
# make a complete set of non purchased items for uid 50

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
param_grid = {'k': np.arange(10,50,10)}

from surprise.model_selection.split import KFold
kfold = KFold(n_splits=5, random_state=2021, shuffle=True)
gs = GridSearchCV(surprise.KNNBasic, param_grid, 
                  measures=['rmse', 'mae'], cv=kfold)

gs.fit(surprise_data)
# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


 








