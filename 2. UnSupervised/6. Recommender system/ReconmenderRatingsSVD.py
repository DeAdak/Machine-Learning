# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:49:48 2022

@author: R
"""
import pandas as pd
import numpy as np
import surprise




rating=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/21. Recommender Systems/filmtrust/ratings.txt"
                   ,sep=" ",names=['uid','iid','rating'])
min_rating=rating['rating'].min()
max_rating=rating['rating'].max()
print(f"rating range is between {min_rating} and {max_rating}")
reader= surprise.Reader(rating_scale=(min_rating,max_rating))
data=surprise.Dataset.load_from_df(rating, reader)
similarity_options = {'name':'cosine','user_based':True}
algo = surprise.SVD(random_state=2022)
output = algo.fit(data.build_full_trainset())
# .fit is going to find the ?, k = 40(default)
pred=algo.predict(uid='53',iid='3')
print(pred.est)

iids=rating['iid'].unique()
iids50 = rating.loc[rating['uid']==50,'iid']
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


 









