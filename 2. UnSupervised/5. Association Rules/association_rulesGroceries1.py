# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:44:01 2022

@author: R
"""

import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries = []
groceries = open("G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/Groceries.csv","r").read()

groceries = groceries.split("\n")


groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
  
te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
te_ary
fp_df = pd.DataFrame(te_ary.astype("int"), columns=te.columns_)


itemFrequency = fp_df.sum(axis=0) / len(fp_df)

# and plot as histogram
ax = itemFrequency.plot.bar(color='blue')
plt.ylabel('Item frequency (relative)')
plt.show()

#################################
itemSets = apriori(fp_df,min_support=0.01,use_colnames=(True),verbose=2)
rules=association_rules(itemSets,metric="confidence",min_threshold=0.5)
rules.sort_values(by=['lift'],ascending=False).head(6)
relv_rules=rules[rules["lift"]>1]


itemSets = apriori(fp_df,min_support=0.05,use_colnames=(True),verbose=2)
rules=association_rules(itemSets,metric="confidence",min_threshold=0.2)
rules.sort_values(by=['lift','confidence'],ascending=False)
