# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 09:45:08 2022

@author: R
"""

import pandas as pd
import matplotlib.pyplot as plt
#pip install mlxtend
from mlxtend.frequent_patterns import association_rules

faceplate = pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/Faceplate.csv',index_col=(0))

# Support of 1-tem freq sets
itemFrequency = faceplate.sum(axis=0) / len(faceplate)

# and plot as histogram
ax = itemFrequency.plot.bar(color='blue')
plt.ylabel('Item frequency (relative)') 
plt.show()

itemSets = apriori(faceplate,min_support=0.3,use_colnames=(True),verbose=2)
#    support       itemsets
# 0      0.6          (Red)
# 1      0.7        (White)
# 2      0.6         (Blue)
# 3      0.4   (Red, White)
# 4      0.4    (Red, Blue)
# 5      0.4  (White, Blue)


rules=association_rules(itemSets,metric="confidence",min_threshold=0.5)
rules.sort_values(by=['lift'],ascending=False)

relv_rules=rules[rules["lift"]>1]
relv_rules.sort_values(by='confidence',ascending=False)
