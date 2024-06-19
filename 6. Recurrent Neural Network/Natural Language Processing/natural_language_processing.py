import os
os.chdir(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_\Datasets")
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t')
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
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
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
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stops)]
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
mystopwrds.add("food")
wc = WordCloud(stopwords=mystopwrds,relative_scaling=1.0,
               background_color="white")
wordcloud = wc.generate(corp_str)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



##################################################

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X,columns=cv.get_feature_names_out())
y = dataset.iloc[:, 1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

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

print(accuracy_score(y_test, y_pred))


####################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X,columns=cv.get_feature_names())
y = dataset.iloc[:, 1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

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

print(accuracy_score(y_test, y_pred))





