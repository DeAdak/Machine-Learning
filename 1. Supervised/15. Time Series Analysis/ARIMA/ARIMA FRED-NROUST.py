import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

fred=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/FRED-NROUST.csv")
y = fred["Value"]
y_train = y[:-12]
y_test = y[-12:]

fred.head()

fred.plot.line(x = 'Date',y = 'Value')
plt.show()

from pmdarima.arima import auto_arima
model = auto_arima(y_train,trace=True,error_action='ignore',suppress_warnings=True)

#Best model:  ARIMA(1,2,0)(0,0,0)[0] 
#Total fit time: 3.610 seconds

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(fred['Value'], lags=20)
plt.show()

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
plt.plot(predictions,label='ARIMA Predicted', color='red')
plt.legend(loc = 'best')
plt.show()

# plot
y_train.plot(label='Train',color="blue")
plt.legend(loc = 'best')
plt.show()
y_test.plot(label='Test',color="pink")
plt.legend(loc = 'best')
plt.show()
predictions.plot(label='ARIMA Prediction',color="purple")
plt.legend(loc = 'best')
plt.show()

y_train.plot(label='Train',color="blue")
y_test.plot(label='Test',color="pink")
predictions.plot(label='ARIMA Prediction',color="purple")
plt.legend(loc = 'best')
plt.show()

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test ARIMA RMSE: %.3f' % rms) #Test RMSE: 0.015


########################## auto_arima ###################################

from pmdarima.arima import auto_arima
model = auto_arima(y_train, trace=True,
                   error_action='ignore', 
                   suppress_warnings=True)
# Best model:  ARIMA(1,2,0)(0,0,0)[0]          
# Total fit time: 2.719 seconds
forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['Prediction'])

# Best model:  ARIMA(1,2,0)(0,0,0)[0]          
# Total fit time: 2.857 seconds

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
print('Test auto_arima RMSE: %.3f' % rms) #Test auto_arima RMSE: 0.002
#Test auto_arima RMSE: 0.002

########################## SARMIA ##########################
model = auto_arima(y_train, trace=True, 
                   error_action='ignore', 
                   suppress_warnings=True,
                   seasonal=True,m=12)

#Best model:  ARIMA(1,2,0)(2,0,0)[12]          
# Total fit time: 18.301 seconds

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['SARMIA'])

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.legend(loc = 'best')
plt.show()
plt.plot(y_test, label='Test',color="pink")
plt.legend(loc = 'best')
plt.show()
plt.plot(forecast, label='SARMIA',color="purple")
plt.legend(loc = 'best')
plt.show()
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Test',color="pink")
plt.plot(forecast, label='SARMIA',color="purple")
plt.legend(loc = 'best')
plt.show()


# plot results
plt.plot(y_test, label='Test',color="pink")
plt.plot(forecast, label='SARMIA',color='red')
plt.legend(loc = 'best')
plt.show()

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test SARMIA RMSE: %.3f' % rms) #Test SARMIA RMSE: 0.003

################# Next 6 Months Prediction ##############
#### Building model on the whole data
model = auto_arima(y, trace=True, error_action='ignore', 
                   suppress_warnings=True)
#Best model:  ARIMA(5,1,3)(0,0,0)[0]     

import numpy as np
forecast = model.predict(n_periods=6)
forecast = pd.DataFrame(forecast,index = np.arange(y.shape[0]+1,y.shape[0]+7),
                        columns=['Prediction'])

# Best model:  ARIMA(2,2,5)(0,0,0)[0]          
# Total fit time: 27.631 seconds

#plot the predictions for validation set

plt.plot(forecast, label='Prediction',color="purple")
plt.legend(loc = 'best')
plt.show()


plt.plot(y, label='Train',color="blue")
plt.plot(forecast, label='Prediction',color="purple")
plt.legend(loc = 'best')
plt.show()

 


