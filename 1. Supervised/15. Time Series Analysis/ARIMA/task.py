# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:47:04 2022

@author: R
"""
##https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/#ProblemStatement
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

tsf=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Time Series Forecasting/Train_SU63ISt.csv",
                header=0,index_col=0)
#tsft=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Time Series Forecasting/Test_0qrQsBZ.csv")
tsf.head()
tsf.shape
tsf.plot.line(x="Datetime",y="Count")


y=tsf["Count"]
y_train = y[:17000]
y_test = y[17000:]

################# ARIMA ####################################

from statsmodels.tsa.arima.model import ARIMA

# train ARIMA
model = ARIMA(y_train,order=(3,1,0))
model_fit = model.fit()
#print('Lag: %s' % model_fit._k_ar)
#print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)

# plot results
plt.plot(y_test,label='Test')
plt.plot(predictions,label='ARIMA', color='red')
plt.legend(loc = 'best')
plt.show()

# plot
y_train.plot(label='Train',color="blue")
plt.legend(loc = 'best')
plt.show()
y_test.plot(label='Test',color="pink")
plt.legend(loc = 'best')
plt.show()
predictions.plot(label='ARIMA',color="purple")
plt.legend(loc = 'best')
plt.show()

y_train.plot(label='Train',color="blue")
y_test.plot(label='Test',color="pink")
predictions.plot(label='ARIMA',color="purple")
plt.legend(loc = 'best')
plt.show()

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test ARIMA RMSE: %.3f' % rms) #Test ARIMA RMSE: 359.673

############################## Auto ARIMA ##################################
from pmdarima.arima import auto_arima
model = auto_arima(y_train,trace=True,
                   error_action='ignore',
                   suppress_warnings=True)

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['auto_arima'])

# Best model:  ARIMA(5,1,0)(0,0,0)[0]          
# Total fit time: 347.352 seconds

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.legend(loc = 'best')
plt.show()
plt.plot(y_test, label='Test',color="pink")
plt.legend(loc = 'best')
plt.show()
plt.plot(forecast, label='auto_arima',color="purple")
plt.legend(loc = 'best')
plt.show()
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Test',color="pink")
plt.plot(forecast, label='auto_arima',color="purple")
plt.legend(loc = 'best')
plt.show()


# plot results
plt.plot(y_test, label='Test',color="pink")
plt.plot(forecast, label='auto_arima',color='red')
plt.legend(loc = 'best')
plt.show()

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test Auto ARIMA RMSE: %.3f' % rms) #Test Auto ARIMA RMSE: 359.406

############################## SARMIA ##############################
model = auto_arima(y_train, trace=True, 
                   error_action='ignore', 
                   suppress_warnings=True,
                   seasonal=True,m=24)

#Best model:  ARIMA(1,2,0)(2,0,0)[12]          
# Total fit time: 18.301 seconds

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['Prediction'])

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.legend(loc = 'best')
plt.show()
plt.plot(y_test, label='Test',color="pink")
plt.legend(loc = 'best')
plt.show()
plt.plot(forecast, label='Prediction',color="purple")
plt.legend(loc = 'best')
plt.show()
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Test',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
plt.legend(loc = 'best')
plt.show()


# plot results
plt.plot(y_test, label='Test',color="pink")
plt.plot(forecast, label='Prediction',color='red')
plt.legend(loc = 'best')
plt.show()

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test SARMIA RMSE: %.3f' % rms)

################# Next 6 Months Prediction ##############
#### Building model on the whole data
model = auto_arima(y, trace=True, error_action='ignore', 
                   suppress_warnings=True)
#Best model:  ARIMA(5,1,3)(0,0,0)[0]     

import numpy as np
forecast = model.predict(n_periods=6)
forecast = pd.DataFrame(forecast,index = np.arange(y.shape[0]+1,y.shape[0]+7),
                        columns=['Prediction'])

#plot the predictions for validation set
plt.plot(y, label='Train',color="blue")

plt.plot(forecast, label='Prediction',color="purple")
plt.show()


#######################  SimpleExpSmoothing  ############################
from statsmodels.tsa.api import SimpleExpSmoothing
alpha = 0.1
fit1=SimpleExpSmoothing(y_train).fit(smoothing_level=alpha,optimized=False)
fcast1 = fit1.forecast(len(y_test))

plt.plot(y_test,label='Test')
plt.plot(fcast1,label="SimpleExpSmoothing")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast1,label="SimpleExpSmoothing")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast1))
print(rms) #301.16625850084716


####################### Holt's Method #######################
### Linear Trend ##
alpha = 0.9
beta = 0.01
from statsmodels.tsa.api import Holt
fit1=Holt(y_train).fit()
fcast1 = fit1.forecast(len(y_test))

plt.plot(y_test,label='Test')
plt.plot(fcast1,label="Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast1,label="Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast1))
print(rms) #3184.2975989724814
### Exponential Trend ##
alpha = 0.9
beta = 0.01
fit1=Holt(y_train,exponential=True).fit()
fcast1 = fit1.forecast(len(y_test))

plt.plot(y_test,label='Test')
plt.plot(fcast1,label="Holt's Exponential Trend")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast1,label="Holt's Exponential Trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast1))
print(rms) 

### Holt Winter's Additive trend Method##
from statsmodels.tsa.api import ExponentialSmoothing
fit4=ExponentialSmoothing(y_train,seasonal_periods=len(y_test),trend='add',seasonal='add').fit()
fcast4 = fit4.forecast(len(y_test))
plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast4,label="Holt Winter's Additive trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast4))
print(rms)  #522.7484639533037

### Holt Winter's Additive and Damped trend Method##
fit4=ExponentialSmoothing(y_train,damped_trend=True,seasonal_periods=len(y_test),trend='add',seasonal='add').fit()
fcast4 = fit4.forecast(len(y_test))

plt.plot(y_test,label='Test')
plt.plot(fcast4,label="Holt Winter's Additive & Damped trend")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast4,label="Holt Winter's Additive & Damped trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast4))
print(rms) #666.3824623659037

### Holt Winter's Multiplicative trend Method##
from statsmodels.tsa.api import ExponentialSmoothing
fit5=ExponentialSmoothing(y_train,seasonal_periods=len(y_test),
                          trend='add',seasonal='mul').fit()
fcast5 = fit5.forecast(len(y_test))

plt.plot(y_test,label='Test')
plt.plot(fcast5,label="Holt Winter's Multiplicative trend")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast5,label="Holt Winter's Multiplicative trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast5))
print(rms) #366.8565034180085

### Holt Winter's Multiplicative and Damped trend Method##
from statsmodels.tsa.api import ExponentialSmoothing
fit5=ExponentialSmoothing(y_train,damped_trend=True,
                          seasonal_periods=len(y_test),trend='add',seasonal='mul').fit()
fcast5 = fit5.forecast(len(y_test))

plt.plot(y_test,label='Test')
plt.plot(fcast5,label="Holt Winter's Multiplicative and Damped trend")
plt.legend(loc='best')
plt.show()

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(fcast5,label="Holt Winter's Multiplicative and Damped trend")
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(y_test, fcast5))
print(rms) #367.73659369261884




