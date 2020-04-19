#importing libraries here

import pandas as pd 
import numpy as np 

import seaborn as sns

#import data

df = pd.read_csv('fake_reg.csv')

X = df[['feature1', 'feature2']].values

y = df['price'].values

from sklearn.model_selection import train_test_split

#spliting data
X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.3,random_state=42)



#scaling sets 
from sklearn.preprocessing import MinMaxScaler

scaler  = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#loading models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation

model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(4))

model.compile(optimizer='rmsprop', loss='mse')


#fitting model to train sets
model.fit(X_train,y_train,epochs=250)


from tensorflow.keras.models import load_model

model.save('my_model.h5')




