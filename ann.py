import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)
 
# onehotencoder = OneHotEncoder(handle_unknown='ignore')
# testX = onehotencoder.fit_transform(X).toarray()
X= X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#---------------DATA PREPROCESSING FINISHED---------------------
# Lets make ANN
import keras
from keras.models import Sequential
#library to create layers
from keras.layers import Dense

# Init Neural Network
classifier = Sequential()

#Adding first hidden layer 
classifier.add(Dense(output_dim=6, init='uniform',activation='relu',input_dim=11))
#Adding Second hidden layer 
classifier.add(Dense(output_dim= 6, init='uniform',activation='relu'))
#Adding the output layer
classifier.add(Dense(output_dim= 1, init='uniform',activation='sigmoid'))
#compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train,y_train,epochs=100,batch_size=10)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)