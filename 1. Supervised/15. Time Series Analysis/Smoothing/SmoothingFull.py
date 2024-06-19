# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 13:08:35 2022

@author: R
"""

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
milk=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/monthly-milk-production-pounds-p.csv")
milk.plot.line(x="Month",y="Milk")

y=milk["Milk"]
y_train = y[:156]
y_test = y[156:]

######## Centered Moving Averge ########## for Visualization 
fcast=y.rolling(3,center=True).mean()
#MA is calculated using df.rolling()
plt.plot(y,label='Data')
plt.legend(loc='best')
plt.show()

plt.plot(fcast,label='Centered Moving Averge Forcast')
plt.legend(loc='best')
plt.show()

plt.plot(y,label='Data')
plt.plot(fcast,label='Centered Moving Averge Forcast')
plt.legend(loc='best')
plt.show()

######## Trailing Moving Averge ########## for Forcasting
span = 10
fcast=y_train.rolling(span).mean()
MA = fcast.iloc[-1] #218550.0
MA_series= pd.Series(MA.repeat(len(y_test)))
MA_Fcast = pd.concat([fcast,MA_series],ignore_index=True)

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()

plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()

plt.plot(MA_Fcast,label="Trailing MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(MA_Fcast,label="Trailing MAForcast")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, MA_series))
print(rms)  # 71.4970861877135   span = 5
            # 54.5962452921443   span = 10

####  SimpleExpSmoothing  ########
from statsmodels.tsa.api import SimpleExpSmoothing
alpha = 0.1 # 0 < alpha <= 1
fit1=SimpleExpSmoothing(y_train).fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test)) #211077

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()

plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()

plt.plot(fcast1,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast1,label="SimpleExpSmoothing")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast1))
print(rms) #57.02407775273818

### Holt's Method##
### Linear Trend ##
alpha = 0.9
beta = 0.01
from statsmodels.tsa.api import Holt
fit1=Holt(y_train).fit(smoothing_level=alpha,smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()

plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()

plt.plot(fcast1,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast1,label="Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast1))
print(rms) #102.13849221483085

### Exponential Trend ##
alpha = 0.9
beta = 0.01
fit1=Holt(y_train,exponential=True).fit(smoothing_level=alpha,smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()
plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()
plt.plot(fcast1,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast1,label="Holt's Exponential Trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast1))
print(rms) #107.76421571185499

### Additive Damped Trend ####
alpha = 0.7
phi = 0.01
fit3=Holt(y_train,damped_trend=True).fit(smoothing_level=alpha,smoothing_slope=phi)
fcast3 = fit3.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()
plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()
plt.plot(fcast3,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast3,label="Additive Damped Trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast3))
print(rms) #84.22578787424938

### Multiplicative Damped Trend ####
alpha = 0.7
phi = 0.01
fit3=Holt(y_train,exponential=True,damped_trend=True).fit(smoothing_level=alpha,smoothing_slope=phi)
fcast3 = fit3.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()
plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()
plt.plot(fcast3,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast3,label="Multiplicative Damped Trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast3))
print(rms) # 82.70267105367338

### Holt Winter's Additive trend Method##
from statsmodels.tsa.api import ExponentialSmoothing
fit4=ExponentialSmoothing(y_train,seasonal_periods=len(y_test),trend='add',seasonal='add').fit()
fcast4 = fit4.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()
plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()
plt.plot(fcast4,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast4,label="Holt Winter's Additive trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast4))
print(rms) # 11.932711508883663

### Holt Winter's Additive and Damped trend Method##
fit4=ExponentialSmoothing(y_train,damped_trend=True,seasonal_periods=len(y_test),trend='add',seasonal='add').fit()
fcast4 = fit4.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()
plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()
plt.plot(fcast4,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast4,label="Additive and Damped trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast4))
print(rms) # 11.753214272071482

### Holt Winter's Multiplicative trend Method##
from statsmodels.tsa.api import ExponentialSmoothing
fit5=ExponentialSmoothing(y_train,seasonal_periods=len(y_test),
                          trend='add',seasonal='mul').fit()
fcast5 = fit5.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()
plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()
plt.plot(fcast5,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast5,label="Holt Winter's Multiplicative trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast5))
print(rms) # 18.137744928718607

### Holt Winter's Multiplicative and Damped trend Method##
from statsmodels.tsa.api import ExponentialSmoothing
fit5=ExponentialSmoothing(y_train,damped_trend=True,
                          seasonal_periods=len(y_test),trend='add',seasonal='mul').fit()
fcast5 = fit5.forecast(len(y_test))

plt.plot(y_train,label='Train')
plt.legend(loc='best')
plt.show()
plt.plot(y_test,label='Test')
plt.legend(loc='best')
plt.show()
plt.plot(fcast5,label="MAForcast")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast5,label="Multiplicative and Damped")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast5))
print(rms) # 17.103479986408885

