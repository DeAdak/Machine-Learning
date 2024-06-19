# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 14:26:01 2022

@author: R
"""
import pandas as pd
import numpy as np
import surprise


jokes=pd.read_excel(r"C:/Users/R/Downloads/12 Practical Machine Learning_/dayWise/Day15/jokes.xlsx",header=None)
jokes1=jokes.iloc[:,1:]
jokes1['uid']=jokes1.index+1
jokes_rating=pd.melt(jokes1,id_vars='uid',var_name='iid',value_name='rating')
rating=jokes_rating[jokes_rating['rating']!=99]
rating.loc[rating["rating"] <=-10.0, "rating"] = 10.0
min_rating=rating['rating'].min()
rating.loc[rating["rating"] >=10.0, "rating"] = 10.0
max_rating=rating['rating'].max()
print(f"rating range is between {min_rating} and {max_rating}")
reader= surprise.Reader(rating_scale=(min_rating,max_rating))
data=surprise.Dataset.load_from_df(rating, reader)
similarity_options = {'name':'cosine','user_based':True}
algo = surprise.KNNBasic(sim_options=similarity_options)
output = algo.fit(data.build_full_trainset())

iids=rating['iid'].unique()
iids100 = rating.loc[rating['uid']==100,'iid']
# iids that are not purchased by uid 50
iids_to_predict = np.setdiff1d(iids,iids100)

testset=[[100,iid,0] for iid in iids_to_predict]
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
