# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:49:59 2022

@author: R
"""

import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

DataSetA = open("G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/DataSetA.csv","r").read()
DataSetA = DataSetA.split("\n")

DataSetA_list = []
for i in DataSetA:
    DataSetA_list.append(i.split(","))
  
te = TransactionEncoder()
te_ary = te.fit(DataSetA_list).transform(DataSetA_list)
te_ary
fp_df = pd.DataFrame(te_ary.astype("int"), columns=te.columns_)


itemFrequency = (fp_df.sum(axis=0) / len(fp_df))
itemFrequency=pd.DataFrame(itemFrequency)
itemFrequency=itemFrequency.iloc[1:,:]
# and plot as histogram
ax = itemFrequency.plot.bar(color='blue')
plt.ylabel('Item frequency (relative)')
plt.show()

fp_df = pd.DataFrame(te_ary.astype("int"), columns=te.columns_).iloc[:,1:]
itemSets = apriori(fp_df,min_support=0.01,use_colnames=(True),verbose=2)
rules=association_rules(itemSets,metric="confidence",min_threshold=0.5)
rules.sort_values(by=['lift'],ascending=False).head(6)
relv_rules=rules[rules["lift"]>1]
relv_rules=relv_rules.sort_values(by='confidence',ascending=False)


itemSets = apriori(fp_df,min_support=0.05,use_colnames=(True),verbose=2)
rules=association_rules(itemSets,metric="confidence",min_threshold=0.5)
rules.sort_values(by=['lift'],ascending=False)
relv_rules=rules[rules["lift"]>1]
relv_rules=relv_rules.sort_values(by=['lift','confidence'],ascending=False)
