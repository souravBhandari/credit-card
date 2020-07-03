import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE # over sampling
from imblearn.under_sampling import NearMiss # under sampling
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

data = pd.read_csv('creditcard.csv')
print(data.shape)

plt.figure(figsize = (10, 8))
sns.countplot(data['Class'])
plt.show()

X = data.drop(labels='Class',axis=1)
Y = data['Class']

SS = StandardScaler()
X['normAmount'] = SS.fit_transform(X['Amount'].values.reshape(-1, 1))
X = X.drop(['Time','Amount'],axis=1)
print(X.head())

np.random.seed(10)
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)

model = Sequential([
    Dense(units=16,input_dim = 29, activation = 'relu'),
    Dense(units=24, activation = 'relu'),
    Dropout(0.5),
    Dense(units=20, activation = 'relu'),
    Dense(units=24, activation = 'relu'),
    Dense(units=1, activation = 'sigmoid'),
])

#model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=15,epochs=5)

#print(model.evaluate(x_test,y_test))

sm = NearMiss(version=2)
X_res , Y_res = sm.fit_resample(X,Y)
print(pd.Series(Y_res).value_counts())
print(X_res.shape,X.shape)

#over sampling
sm =SMOTE()
X_res_OS , Y_res_OS = sm.fit_resample(X,Y)
print(pd.Series(Y_res_OS).value_counts())

np.random.seed(10)
x_train,x_test,y_train,y_test = train_test_split(X_res_OS,Y_res_OS, test_size = 0.2)
print(x_train.shape)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=15,epochs=1,validation_data=[x_test,y_test])
print(model.evaluate(x_test,y_test))
