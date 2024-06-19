


import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
zillow = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/ZILLOW-M1301_MLPSF.csv")
zillow.plot.line(x="Date",y="Value")

y=zillow["Value"]
y_train = y[:-7]
y_test = y[-7:]

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
span = 5
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
print(rms)  # 14488.20949205644   span = 5

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
print(rms) #14407.366728186038

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
print(rms) #57491.12940912842

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
print(rms) #71477.02773452482

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
print(rms) #37973.47491949462

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
print(rms) # 45393.028899554236

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
print(rms) # 19656.159138011317

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
print(rms) # 20804.21313730357

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
print(rms) # 19658.72915527688

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
print(rms) # 38881.68645933577

