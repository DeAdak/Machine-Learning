# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:52:40 2024

@author: R
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Importing the dataset
dataset = pd.read_json(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/dayWise/Natural Language Processing/Sarcasm_Headlines_Dataset_v2.json',lines=True)
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
dataset['headline'][i]
review = re.sub('[^a-zA-Z]', ' ', dataset['headline'][i])
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
    review = re.sub('[^a-zA-Z]', ' ', dataset['headline'][i])
    #print(i)
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
plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis("off") 
plt.show()
    
### to customize the stopwords ##
from wordcloud import STOPWORDS
mystopwrds = set(STOPWORDS)
mystopwrds.add('man')
mystopwrds.add('new')
wc = WordCloud(stopwords=mystopwrds,relative_scaling=1.0,
               background_color="white")
wordcloud = wc.generate(corp_str)
plt.imshow(wordcloud)
plt.figure(figsize=(10,6))
plt.axis("off")
plt.show()

###################
'http://abc.com'.split('/') #['http:', '', 'abc.com']

'http://abc.com'.split('/')[2] #'abc.com'

'http://abc.com'.split('/')[2].split('.') #['abc', 'com']

'http://abc.com'.split('/')[2].split('.')[0] #'abc'

links=[]
i = 5
link=dataset['article_link'][i] #'https://www.huffingtonpost.com/entry/my-white-inheritance_us_59230747e4b07617ae4cbe1a'
link.split('/')[2]
link.split('/')[2].split('.')[1]
links.append(link.split('/')[2].split('.')[1])
#################

links=[]
for i in range(dataset.shape[0]):
    link=dataset['article_link'][i]
    links.append(link.split('/')[2].split('.')[1])
    
le = LabelEncoder()
links_en = le.fit_transform(links)

links_pd=pd.DataFrame(links_en,columns=['site'])

dataset_pd = pd.concat([dataset,links_pd],axis=1)

###########################   CountVectorizer  ##############################
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X,columns=cv.get_feature_names_out())
y = dataset.iloc[:, 1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2024,stratify=y)


#########   GaussianNB   ############
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred)) #0.7582

#########   RandomForestClassifier   ############
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV
rfc=RandomForestClassifier(random_state=2024)
param={'max_features':[5,10,15,20]}
kfold=KFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=GridSearchCV(rfc, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) #0.9162853385930309
print(cv.best_params_) #{'max_features': 5}

#############  DecisionTreeClassifier  #############
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2024,
                        shuffle=True)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=2024)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc',verbose=2)

cv.fit(X,y)

# Best Parameters
print(cv.best_params_) # {'max_depth': 4, 'min_samples_leaf': 10, 'min_samples_split': 5}

print(cv.best_score_) #0.9859106805646292

#############  XGBRFClassifier #############
from sklearn.model_selection import KFold,GridSearchCV
from xgboost import XGBRFClassifier
xgbr=XGBRFRegressor(random_state=2024)
lr=[0.01,0.1,0.3,0.5,0.6]
n_est=[10,25,50]
max_d=[3,5,10]
params=dict(learning_rate=lr,max_depth=max_d,n_estimators=n_est)
kfold=StratifiedKFold(n_splits=(5),shuffle=(True),random_state=(2024))
cv=GridSearchCV(xgbr, param_grid=params,scoring='roc_auc',verbose=2,cv=kfold)
cv.fit(X,y)
cv.best_score_ #0.7489972996983496
cv.best_params_ #{'learning_rate': 0.6, 'max_depth': 10, 'n_estimators': 10}



###########################   TfidfVectorizer   ##############################
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X,columns=cv.get_feature_names_out())
y = dataset.iloc[:, 1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#########   GaussianNB   ############
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred)) #0.8122

#########   RandomForestRegressor   ############
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,GridSearchCV
rfr=RandomForestRegressor(random_state=2024,verbose=3)
param={'max_features':[6,10,15,20]}
kfold=KFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=GridSearchCV(rfr, param_grid=param,scoring='r2',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) #0.8747915836339892
print(cv.best_params_) #{'max_features': 15}

#########   RandomForestClassifier   ############
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV
rfc=RandomForestClassifier(random_state=2024)
param={'max_features':[5,10,15,20]}
kfold=KFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=GridSearchCV(rfc, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) #0.9162853385930309
print(cv.best_params_) #{'max_features': 5}

#############  DecisionTreeClassifier  #############
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2024,
                        shuffle=True)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=2024)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc',verbose=2)

cv.fit(X,y)

# Best Parameters
print(cv.best_params_) # {'max_depth': 4, 'min_samples_leaf': 10, 'min_samples_split': 5}

print(cv.best_score_) #0.9859106805646292

#############  XGBRFClassifier #############
from sklearn.model_selection import KFold,GridSearchCV
from xgboost import XGBRFClassifier
xgbr=XGBRFRegressor(random_state=2024)
lr=[0.01,0.1,0.3,0.5,0.6]
n_est=[10,25,50]
max_d=[3,5,10]
params=dict(learning_rate=lr,max_depth=max_d,n_estimators=n_est)
kfold=StratifiedKFold(n_splits=(5),shuffle=(True),random_state=(2024))
cv=GridSearchCV(xgbr, param_grid=params,scoring='roc_auc',verbose=2,cv=kfold)
cv.fit(X,y)
cv.best_score_ #0.7489972996983496
cv.best_params_ #{'learning_rate': 0.6, 'max_depth': 10, 'n_estimators': 10}


    



    
    
    
    
    
    

