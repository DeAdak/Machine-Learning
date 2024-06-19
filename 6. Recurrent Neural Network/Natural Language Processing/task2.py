# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:49:21 2024

@author: R
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:44:50 2024

@author: R
"""



import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/dayWise/Natural Language Processing/labeledTrainData.tsv', sep = '\t')
dataset.shape
dataset.columns
# Cleaning the texts
import re
#pip install nltk
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stops = stopwords.words('english')

from nltk.stem.porter import PorterStemmer

###############Understanding the loop############
#=============================================================================
i = 6 #Honeslty it didn't taste THAT fresh.)
dataset['review'][i]
review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
#Honeslty it didn t taste THAT fresh  
review = review.lower() #honeslty it didn t taste that fresh  
review = review.split() 
review #['honeslty', 'it', 'didn', 't', 'taste', 'that', 'fresh']
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stops)]
review #['honeslti', 'tast', 'fresh']
review = ' '.join(review)
review
corpus = []
corpus.append(review)
#=============================================================================
#################################################

corpus = []
for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
    print(i)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stops)]
    review = ' '.join(review)
    corpus.append(review)

# Generating Word Cloud
#pip install wordcloud
corp_str = str(corpus)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(relative_scaling=1.0).generate(corp_str)
plt.imshow(wordcloud)
plt.axis("off") 
plt.show()
    
### to customize the stopwords ##
from wordcloud import STOPWORDS
mystopwrds = set(STOPWORDS)
mystopwrds.add("br")
wc = WordCloud(stopwords=mystopwrds,relative_scaling=1.0,
               background_color="white")
wordcloud = wc.generate(corp_str)
plt.imshow(wordcloud)
plt.figure(figsize=(10,6))
plt.axis("off")
plt.show()

###########################   CountVectorizer  ##############################
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X,columns=cv.get_feature_names_out())
y = dataset.iloc[:, 1]


################  Principle Component Analysis  ###############
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2024,stratify=y)
stdscl=StandardScaler()
X_train_scaled=stdscl.fit_transform(X_train)
pca=PCA()
principleComponents = pca.fit_transform(X_train_scaled)
principleFs=pd.DataFrame(principleComponents[:,:1200])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100
np.cumsum(pca.explained_variance_ratio_ * 100)

X_test_scaled=stdscl.transform(X_test)
X_test_pca=pca.transform(X_test_scaled)
X_test_pca_pd=pd.DataFrame(X_test_pca[:,:1200])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100


ys=pca.explained_variance_ratio_ * 100
xs=np.arange(1,1501)
plt.plot(xs,ys)
plt.show()

ys=np.cumsum(pca.explained_variance_ratio_ * 100)
#ys[1200]
xs=np.arange(1,1201)
plt.plot(xs,ys[:1200])
plt.show()
# 1200 Fs are contributing more than 90%, so we can ditch the rest Fs

################### Plotting the PCs ######################
# df_PC = pd.DataFrame(principleComponents[:,:2],
#                      index=X_train.index,
#                      columns=["PC1","PC2"])
# df_PC = pd.concat([df_PC,y_train],axis=1)


# sns.scatterplot(x="PC1", y="PC2", hue='D',data=df_PC)
# plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
log_reg=LogisticRegression()
log_reg.fit(principleFs,y_train)
y_pred=log_reg.predict(X_test_pca_pd)
print(accuracy_score(y_test, y_pred)) #0.8606666666666667
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred)) #0.8606666666666667

###########################   TfidfVectorizer   ##############################
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X,columns=cv.get_feature_names_out())
y = dataset.iloc[:, 1]


################  Principle Component Analysis  ###############
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2024,stratify=y)
stdscl=StandardScaler()
X_train_scaled=stdscl.fit_transform(X_train)
pca=PCA()
principleComponents = pca.fit_transform(X_train_scaled)
principleFs=pd.DataFrame(principleComponents[:,:1200])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100
np.cumsum(pca.explained_variance_ratio_ * 100)

X_test_scaled=stdscl.transform(X_test)
X_test_pca=pca.transform(X_test_scaled)
X_test_pca_pd=pd.DataFrame(X_test_pca[:,:1200])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100


ys=pca.explained_variance_ratio_ * 100
xs=np.arange(1,1501)
plt.plot(xs,ys)
plt.show()

ys=np.cumsum(pca.explained_variance_ratio_ * 100)
#ys[1200]
xs=np.arange(1,1201)
plt.plot(xs,ys[:1200])
plt.show()
# 1200 Fs are contributing more than 90%, so we can ditch the rest Fs

################### Plotting the PCs ######################
# df_PC = pd.DataFrame(principleComponents[:,:2],
#                      index=X_train.index,
#                      columns=["PC1","PC2"])
# df_PC = pd.concat([df_PC,y_train],axis=1)


# sns.scatterplot(x="PC1", y="PC2", hue='D',data=df_PC)
# plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
log_reg=LogisticRegression()
log_reg.fit(principleFs,y_train)
y_pred=log_reg.predict(X_test_pca_pd)
print(accuracy_score(y_test, y_pred)) #0.8649333333333333
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred)) #0.8649333333333333
