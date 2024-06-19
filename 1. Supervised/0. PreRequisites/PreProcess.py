import pandas as pd
telecom_df = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_\Cases\Telecom\Telecom.csv")
telecom = pd.get_dummies(telecom_df,drop_first=True)

##OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop = 'first')
telecom_ohe = ohe.fit_transform(telecom_df).toarray()
print(type(telecom),type(telecom_ohe))
print(ohe.categories_)

#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import OneHotEncoder
iris = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Datasets/iris.csv")

dum_iris = pd.get_dummies(iris)
dum_iris = pd.get_dummies(iris,drop_first=True)

cars = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Datasets/Cars93.csv")

cars.head(10)

cars = cars.set_index('Model')
# OR
cars.set_index('Model',inplace=True)

dum_cars = pd.get_dummies(cars, drop_first=True)

dum_cars.head(n=10)

## Label Encoding
from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()
y = ['a','b','a','a','c','a','b','b','a','c','a']

trny = lbcode.fit(y)
trny = lbcode.transform(y)
print(trny)
print(lbcode.inverse_transform(trny))

carsMissing = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Datasets/Cars93Missing.csv",index_col=1)

carsMissing.shape

carsDropNA = carsMissing.dropna()
carsDropNA.shape

# Dummying the data
dum_cars_miss = pd.get_dummies(cars, drop_first=True)

job = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Datasets/JobSalary2.csv")
mu_comp = job['Computer'].mean()
job['a_Computer'] = job['Computer'].fillna(mu_comp)
job
job['Computer'].fillna(mu_comp,inplace = True)
job

from sklearn.impute import SimpleImputer
####### constant imputation #######
imp = SimpleImputer(strategy='constant',fill_value = 10)
imp_job = imp.fit_transform(job)
pd_job = pd.DataFrame(imp_job,columns=job.columns)

###### mean imputation ###########
imp = SimpleImputer(strategy='mean')
imp.fit(job)
imp_job = imp.transform(job)
#OR
imp_job = imp.fit_transform(job)

pd_job = pd.DataFrame(imp_job,columns=job.columns)


carsImputed = imp.fit_transform(dum_cars_miss)

df_carsImputed = pd.DataFrame(carsImputed,
                              columns= dum_cars_miss.columns,
                              index=dum_cars_miss.index)

dum_cars_miss.shape
carsImputed.shape
df_carsImputed.shape

###### median imputation ###########

impJob = SimpleImputer(strategy='median')
trn_job = impJob.fit_transform(job)
pd_job = pd.DataFrame(trn_job,columns=job.columns)


import numpy as np
milk = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Datasets/milk.csv",index_col=0)
milk.head()
np.mean(milk), np.std(milk)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(milk)
milkscaled=scaler.transform(milk)
# OR
milkscaled=scaler.fit_transform(milk)


np.mean(milkscaled[:,0]), np.std(milkscaled[:,0])
np.mean(milkscaled[:,1]), np.std(milkscaled[:,1])
np.mean(milkscaled[:,2]), np.std(milkscaled[:,2])
np.mean(milkscaled[:,3]), np.std(milkscaled[:,3])
np.mean(milkscaled[:,4]), np.std(milkscaled[:,4])

# Converting numpy array to pandas
df_milk = pd.DataFrame(milkscaled,columns=milk.columns,
                       index=milk.index)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(milk)
minmaxMilk = minmax.transform(milk)
minmaxMilk[1:5,]

# OR
minmaxMilk = minmax.fit_transform(milk)
# Converting numpy array to pandas
df_milk = pd.DataFrame(minmaxMilk,columns=milk.columns,
                       index=milk.index)