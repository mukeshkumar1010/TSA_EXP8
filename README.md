## Devloped by: MUKESH KUMAR S
## Register Number: 212223240099
## Date: 06-05-2025

# Ex.No: 08     MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file.
3. Display the shape and the first 10 rows of the dataset
4. Perform rolling average transformation with a window size of 5 and 10 
5. Display first 10 and 20 values repecively and plot them both
6. Perform exponential smoothing and plot the fitted graph and orginal graph

### PROGRAM:

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```
Read the AirPassengers dataset
```py
data = pd.read_csv('AirPassengers.csv')
```
Focus on the '#Passengers' column
```py
passengers_data = data[['#Passengers']]
```
Display the shape and the first 10 rows of the dataset
```py
print("Shape of the dataset:", passengers_data.shape)
print("First 10 rows of the dataset:")
print(passengers_data.head(10))
```
Plot Original Dataset (#Passengers Data)
```py
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original #Passengers Data')
plt.title('Original Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
```
Moving Average
Perform rolling average transformation with a window size of 5 and 10
```py
rolling_mean_5 = passengers_data['#Passengers'].rolling(window=5).mean()
rolling_mean_10 = passengers_data['#Passengers'].rolling(window=10).mean()
```
Display the first 10 and 20 vales of rolling means with window sizes 5 and 10 respectively
```py
rolling_mean_5.head(10)
rolling_mean_10.head(20)
```
Plot Moving Average
```py
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
```
Perform data transformation to better fit the model
```py
data_monthly = data.resample('MS').sum()   #Month start
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),index=data.index)

```
Exponential Smoothing
```py
# The data seems to have additive trend and multiplicative seasonality
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, yes even zeros
x=int(len(scaled_data)*0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()
```
Make predictions for one fourth of the data
```py
model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
predictions.plot(ax=ax)
ax.legend(["data_monthly", "predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')

```

### OUTPUT:

Original data:

![image](https://github.com/user-attachments/assets/d41e32c3-9be8-4dc5-906f-77284a68d981)

![image](https://github.com/user-attachments/assets/7d02b0ad-7e6f-4a72-9dc0-89637fcd07a8)


Moving Average:- (Rolling)

window(5):

![image](https://github.com/user-attachments/assets/9956d549-8d51-4dc6-94f9-2fbe0c96d34f)


window(10):


![image](https://github.com/user-attachments/assets/6f1580a0-e105-44cc-94ec-35b49c322246)

plot:

![image](https://github.com/user-attachments/assets/19970323-6694-440a-a4c9-16227065efbc)


Exponential Smoothing:-

Test:

![image](https://github.com/user-attachments/assets/bf715e75-1e70-4bac-b2fa-9ec0c30da211)


Performance: (MSE)

![image](https://github.com/user-attachments/assets/9b150edb-6dc6-407b-8d2b-212ec95e0b2f)


Prediction:

![image](https://github.com/user-attachments/assets/ec46b79d-d8b4-42b6-917f-2b4c71591e65)




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
