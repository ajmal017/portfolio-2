import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


class StockNet:
   def __init__(self, data_file, test_size=0.2, time=50, offset=10, sample_spacing=5):
      datafile = open(data_file, "rb")
      self.database = pd.read_csv(open(data_file, "rb"))
      test_index =  int((1-test_size)*len(self.database))
      self.train_data = self.database[0:test_index].copy()
      self.test_data = self.database[test_index:len(self.database)].copy()

      self.test_size  = test_size
      self.train_size = 1 - test_size
      self.size = len(self.database['Open'])
      self.__standardize()
      self.prepare_dataset(time, offset) 

      self.model = keras.models.Sequential()
      self.model.add(LSTM(time*5, activation=tf.nn.sigmoid, input_shape=(time//sample_spacing+1, 1)))
      self.model.add(Dense(1))
      self.model.compile(optimizer='adam', loss = 'mse')

      # Initilizing stats
      self.vol_close_R = scipy.stats.pearsonr(self.database['Volume'], self.database['Close'])
      
   def __standardize(self):
      # Means and Standard Deviations only calculated using training data
      train_len = self.train_size*self.size
      close_mean, close_std = self.database.loc[0:int(train_len):,'Close'].mean(),\
      self.database.loc[0:int(train_len):,'Close'].std()
      vol_mean, vol_std     = self.database.loc[0:int(train_len):,'Volume'].mean(),\
      self.database.loc[0:int(train_len):,'Volume'].std()

      # TODO: Add separate class told hold invidual stats of databases
      self.close_mean, self.close_std = close_mean, close_std
      self.vol_mean, self.vol_std = vol_mean, vol_std

      
      # Standardizing data
      self.database.loc[:,'Close']     =\
      np.divide(np.subtract(self.database['Close'].to_numpy(), close_mean), close_std)
      self.database.loc[:,'Volume']    =\
      np.divide(np.subtract(self.database['Volume'].to_numpy(), vol_mean), vol_std)


   # Prepares training/testing data, based on the input time and offset
   def prepare_dataset(self, time=25, offset=10, sample_spacing=5):
      self.close_train_data = []
      self.close_train_labels = []
      self.close_test_data = []
      self.close_test_labels = []
      changes = []

      
      for i in range(time,int(self.train_size*self.size)-offset):
         self.close_train_data.append(self.database.loc[(i-time):i:sample_spacing,'Close'].to_numpy())
         self.close_train_labels.append(self.database.loc[i+offset-1,'Close'])
         changes.append(np.abs(self.database.loc[i+offset-1,'Close'] - self.database.loc[i,'Close']))

      for i in range(int(self.train_size*self.size)+time, self.size-offset):
         self.close_test_data.append(self.database.loc[(i-time):i:sample_spacing,'Close'].to_numpy())
         self.close_test_labels.append(self.database.loc[i+offset-1,'Close'])
 

      print(f"=======\nAverage Change over {offset} days : {np.mean(changes)}\n=======") 
      self.close_train_data = np.array(self.close_train_data) 
      self.close_test_data = np.array(self.close_test_data) 
      self.close_train_labels = np.array(self.close_train_labels) 
      self.close_test_labels = np.array(self.close_test_labels) 
      

      self.close_train_data = self.close_train_data.reshape(\
      tuple(list(np.shape(self.close_train_data)) + [1]))
      self.close_test_data = self.close_test_data.reshape(\
      tuple(list(np.shape(self.close_test_data)) + [1]))
   
      

time = 50
offset = 10
sample_spacing = 5
net = StockNet("./database/RIGL.csv", time=time, offset=offset, sample_spacing=sample_spacing)

for i in net.database['Close']:
   if (i == np.nan):
      print("NaN in Close")


for i in net.database['Volume']:
   if (i == np.nan):
      print("NaN in Volume")

net.model.fit(net.close_train_data, net.close_train_labels, epochs=10, verbose=2)
predictions = []
for prediction in net.model.predict(net.close_train_data):
   predictions.append(prediction)
for prediction in net.model.predict(net.close_test_data):
   predictions.append(prediction) 

#predictions = np.multiply(predictions, self.close_std)
#predictions = np.add(predictions, self.close_mean)
 
x_ax = np.linspace(1, len(net.database['Close']), len(net.database['Close']), endpoint=True)
plt.figure()
plt.plot(x_ax, net.database['Close'], color='red', label="Actual price")
x_ax1 = np.linspace(time+offset+1, len(net.close_train_labels)+time+offset,\
len(net.close_train_labels), endpoint=True)
x_ax2 = np.linspace(time+offset+1+len(net.close_train_labels), len(predictions)+2*(time+offset),\
len(net.close_test_labels), endpoint=True)
plt.plot(x_ax1, predictions[0:len(net.close_train_labels):], color='blue', label="Predictions")
plt.plot(x_ax2, predictions[len(net.close_train_labels)::], color='blue', label="Predictions")
plt.legend()
plt.show()

print(f"Actual End: {x_ax[len(x_ax)-1]}")
print(f"Prediction End: {x_ax2[len(x_ax2)-1]}")
