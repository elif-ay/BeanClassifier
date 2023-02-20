# Import the required libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# Read the CSV file:
data = pd.read_csv('C:\\Users\\Elif\\Desktop\\bitirmeProjesi\\dataset.csv')

# Consider features we want to work on:

X = data[[ 'Contrast1', 'Contrast2', 'Contrast3', 'Contrast4', 'Dissimilarity1', 'Dissimilarity2', 'Dissimilarity3', 'Dissimilarity4', 'Homogeneity1', 'Homogeneity2', 'Homogeneity3',
              'Homogeneity4', 'Energy1', 'Energy2', 'Energy3', 'Energy4', 'Correlation1', 'Correlation2', 'Correlation3', 'Correlation4', 'ASM1', 'ASM2', 'ASM3', 'ASM4', 'ORB']]
y = data['SIFT']
'''
import matplotlib.pyplot as plt
plt.scatter(X['Contrast1'], X['ASM1'], c=y)
plt.show()
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 109)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 109)

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("F1:",f1_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",recall_score(y_test, y_pred))

'''

'''

# Generating training and testing data from our data:
# We are using 80% data for training.
train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]
#Modeling:
#Using sklearn package to model data :
regr = linear_model.LinearRegression()
train_x = np.array(train[[ 'Contrast1', 'Contrast2', 'Contrast3', 'Contrast4', 'Dissimilarity1', 'Dissimilarity2', 'Dissimilarity3', 'Dissimilarity4', 'Homogeneity1', 'Homogeneity2', 'Homogeneity3',
              'Homogeneity4', 'Energy1', 'Energy2', 'Energy3', 'Energy4', 'Correlation1', 'Correlation2', 'Correlation3', 'Correlation4', 'ASM1', 'ASM2', 'ASM3', 'ASM4', 'ORB']])
train_y = np.array(train['SIFT'])
regr.fit(train_x,train_y)
test_x = np.array(test[[ 'Contrast1', 'Contrast2', 'Contrast3', 'Contrast4', 'Dissimilarity1', 'Dissimilarity2', 'Dissimilarity3', 'Dissimilarity4', 'Homogeneity1', 'Homogeneity2', 'Homogeneity3',
              'Homogeneity4', 'Energy1', 'Energy2', 'Energy3', 'Energy4', 'Correlation1', 'Correlation2', 'Correlation3', 'Correlation4', 'ASM1', 'ASM2', 'ASM3', 'ASM4', 'ORB']])
test_y = np.array(test['SIFT'])
# print the coefficient values:
coeff_data = pd.DataFrame(regr.coef_ , X.columns , columns=['Coefficients'])
print(coeff_data)
#Now let’s do prediction of data:
Y_pred = regr.predict(test_x)
# Check accuracy:
from sklearn.metrics import r2_score
R = r2_score(test_y , Y_pred)
print ("R² :",R)

'''
