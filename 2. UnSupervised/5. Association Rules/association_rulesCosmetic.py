# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:45:29 2022

@author: R
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

cosmetics_df = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/Cosmetics.csv",index_col=(0))

# Support of 1-tem freq sets
itemFrequency = cosmetics_df.sum(axis=0) / len(cosmetics_df)

# and plot as histogram
ax = itemFrequency.plot.bar(color='blue')
plt.ylabel('Item frequency (relative)') 
plt.show()

itemSets = apriori(cosmetics_df,min_support=0.3,use_colnames=(True),verbose=2)
#    support                 itemsets
# 0    0.363                  (Blush)
# 1    0.442              (Concealer)
# 2    0.357                (Mascara)
# 3    0.381             (Eye shadow)
# 4    0.536             (Foundation)
# 5    0.490              (Lip Gloss)
# 6    0.322               (Lipstick)
# 7    0.457               (Eyeliner)
# 8    0.321    (Eye shadow, Mascara)
# 9    0.356  (Lip Gloss, Foundation)

itemSets = apriori(cosmetics_df,min_support=0.1,use_colnames=(True),verbose=2)
rules=association_rules(itemSets,metric="confidence",min_threshold=0.7)
rules.sort_values(by=['lift','confidence'],ascending=False)
relv_rules=rules[rules["lift"]>1]
relv_rules.sort_values(by=['lift','confidence'],ascending=False).head(3)
rules["lift"]

itemSets = apriori(cosmetics_df,min_support=0.05,use_colnames=(True),verbose=2)
rules=association_rules(itemSets,metric="confidence",min_threshold=0.7)
rules.sort_values(by=['lift','confidence'],ascending=False)
relv_rules=rules[rules["lift"]>1]
relv_rules.sort_values(by=['lift','confidence'],ascending=False).head(3)
rules["lift"]
