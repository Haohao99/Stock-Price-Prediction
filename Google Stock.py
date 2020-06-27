#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Many Thanks for https://www.youtube.com/watch?v=QIUxPv5PJOY&t=326s


# In[2]:


#import packge
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[3]:


#Get the Stock quote
df = web.DataReader('GOOGL',data_source='yahoo',start='2010-01-01',end='2020-04-27')
#show the data
df.head(3)


# In[4]:


#show the shape of the data
df.shape


# In[5]:


#Visualize the colsing price history of Google
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()


# In[6]:


#Create a new dataframe with the "Close" column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*0.8)

training_data_len


# In[7]:


#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[8]:


#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#Split the data into x_train and y_trian data sets
x_train = []
y_train = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()


# In[9]:


#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[10]:


#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[11]:


#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


# In[12]:


#Compilb the model
model.compile(optimizer='adam', loss ='mean_squared_error')


# In[13]:


#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[14]:


#Create the testing data set
#Create a new array containing scaled values from 
test_data = scaled_data[training_data_len - 60:, :]
#Create the data set x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[15]:


#Convert the data to a numpy array
x_test = np.array(x_test)


# In[16]:


#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[17]:


#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[18]:


#Get the root mean squared error(RMSE)
rmse =np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[19]:


#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Prediction'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD (S)',fontsize=18)
plt.plot(valid[['Close','Prediction']])
plt.legend(['Train','Val','Predction'],loc = 'low right')
plt.show()


# In[20]:


valid


# In[21]:


#Get the qutoe
apple_quote = web.DataReader('GOOGL', data_source='yahoo',start='2010-01-01',end='2020-04-27')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#Get teh last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert tje X_test data set tp a numpy array
X_test = np.array(X_test)
#Reshape the data
pred_price = model.predict(X_test)
#udo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[22]:


#Get the qutoe
apple_quote2 = web.DataReader('GOOGL', data_source='yahoo',start='2020-04-28',end='2020-04-28')
print(apple_quote2['Close'])


# In[ ]:





# In[ ]:




