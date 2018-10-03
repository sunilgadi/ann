import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import theano
import os
print(os.listdir("../input"))

dataset = pd.read_csv('../input/Churn_Modelling.csv')
dataset.head(10)

X = dataset.iloc[:, 3:13].values # RowNumber , CustomerId and Surname removed
y = dataset.iloc[:, -1].values # Target Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x_1 = LabelEncoder()
X[:,1] = label_encoder_x_1.fit_transform(X[:,1]) # Encoding Geography variable
label_encoder_x_2 = LabelEncoder()
X[:,2] = label_encoder_x_2.fit_transform(X[:,2])# Encoding Gender Variable
onehotencoder = OneHotEncoder(categorical_features=[1]) # Here 1 indicatest that it is 1st column (column index starting from 0)
# 1st column is Geography
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:] # remove 1st column to avoid dummy variable trap

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)
X_train.shape

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # We need to fit the model using X_train first and then transform it. Hence fit_tranform is used
X_test = sc.transform(X_test) # As it is already fit by X_train, we just need to transform it. Hence transform is used.

import keras
from keras.models import Sequential  # Required to initialize the neural network
from keras.layers import Dense  # Required to build the layers in the neural network

classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' , input_dim = 11)) # creating input and 1st hidden layer.

classifier.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'relu')) # creating 2nd hidden layer.

classifier.add(Dense(output_dim = 1 , init = 'uniform' , activation = 'sigmoid')) # creating output layer.

classifier.compile(optimizer='adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)
