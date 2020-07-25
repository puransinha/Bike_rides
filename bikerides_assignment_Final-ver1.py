# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:22:06 2020

@author: Puran Prakash Sinha
"""

# Assignment - BIke Sharing

# import all the relevent packages & Version checks
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import csv
from time import time


# print the versions of the relevanrt libraries
print('Welcome to Python : {}'.format(sys.version))
print('Version of Numpy is : {}'.format(np.__version__))
print('Version of Pandas is : {}'.format(pd.__version__))
print('Version of SK-Learn is : {}'.format(sklearn.__version__))
print('Version of Seaborn is : {}'.format(sns.__version__))


#Import the Datasets .... Understanding the datasets
bike=pd.read_csv(r'D:\Bike_sharing_assignment\bike-sharing\day.csv')
bike.head(10)
bike.shape
bike.describe()
bike.columns
bike.isnull().sum()
bike.dtypes # Change datatype 'dteday' from object to date time
bike.dteday=pd.to_datetime(bike['dteday'])
bike.dtypes

# ************** Plotting the plot to check *****************

sns.distplot(bike['hum'])
plt.show()

sns.distplot(bike['temp'])
plt.show()

sns.distplot(bike['windspeed'])
plt.show()

# ***************** Removing Irelevant Columns *************
bike.drop(['instant', 'hum', 'temp', 'atemp', 'windspeed'], inplace=True, axis=1)
bike.drop(['dteday'], axis=1, inplace=True)

# ***************** Plotting the graph with existing Datasets before fitting **********


ax = plt.figure(figsize = (10,10))
correlation_matrix = bike.corr().round(2)
sns.heatmap(data=correlation_matrix,cmap="YlGnBu" ,annot=True)

plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(x='season', y='cnt', data=bike)
plt.subplot(2,3,2)
sns.boxplot(x='weekday', y='cnt',data=bike)
plt.subplot(2,3,3)
sns.boxplot(x='mnth',y='cnt',data=bike)
plt.subplot(2,3,4)
sns.boxplot(x='holiday',y='cnt',data=bike)
plt.subplot(2,3,5)
sns.boxplot(x='weathersit',y='cnt',data=bike)
plt.subplot(2,3,6)
sns.boxplot(x='workingday',y='cnt',data=bike)
plt.show()

# We can also visualise some of these categorical features parallely by using the `hue` argument. Below is the plot for `furnishingstatus` with `airconditioning` as the hue.

plt.figure(figsize = (20, 15))
plt.subplot(2,3,1)
sns.boxplot(x = 'holiday', y = 'cnt', hue = 'season', data = bike)
plt.subplot(2,3,2)
sns.boxplot(x = 'weekday', y = 'cnt', hue = 'season', data = bike)
plt.subplot(2,3,3)
sns.boxplot(x = 'workingday', y = 'cnt', hue = 'mnth', data = bike)
plt.subplot(2,3,4)
sns.boxplot(x = 'weathersit', y = 'cnt', hue = 'season', data = bike)
plt.subplot(2,3,5)
sns.boxplot(x='mnth', y= 'cnt', hue = 'yr', data=bike)
plt.show()

#********************************Data Preperations*****************************
from sklearn.model_selection import train_test_split as tts

X = bike.iloc[:, :-1]
y = bike.iloc[:,-1]

X_train , X_test, y_train, y_test = tts(X, y, test_size = 0.3,random_state = 0)

model_scores={}
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

# ********************Checking R2 Square ***************************************

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,RANSACRegressor,Ridge,Lasso
lin_reg= LinearRegression()
rid_reg = Ridge()
las_reg = Lasso()
ran_reg=RANSACRegressor()

scorelist=[]
modeln=[lin_reg,rid_reg,las_reg,ran_reg]

# coefficients in linear model
coef_dict = {}
for coef, feat in zip(lin_reg.coef_,X.columns):
    coef_dict[feat] = coef
coef_dict

######## *********Ordinary Least Square *******************************
print ('\nRunning OLS...\n')
#import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()
lr.params
print(lr.summary())

for i in modeln:
  i.fit(X_train,y_train)
  y_pred=i.predict(X_test)
  scorelist.append(r2_score(y_test,y_pred))

print(scorelist) #Not much significant difference between ridge and linear regression
model_scores.update(multiple_reg = scorelist[0])
print(model_scores)

plt.figure(figsize=[6,6])
plt.scatter(X_train, X_train)
plt.show()

y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)
fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()

# ************************** Decision Tree *************************
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state = 0)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
dt_score = r2_score(y_test,y_pred)
model_scores.update(decision_tree = dt_score)

# ************************* Random Forest **************************
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 0)
model.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model , X = X_train , y = y_train , cv = 10)
rfscore = accuracies.mean()*100
print("Accuracy {:.2f} %".format(rfscore))
print("Standard deviation {:.2f} %".format(accuracies.std()*100))
model_scores.update(random_forest = rfscore)
model_scores

# *********************** Tuning the Hyperparameters *********************
from sklearn.model_selection import GridSearchCV
params_rf = {
'n_estimators': [10,50,100,200],
'max_features': ["auto","log2","sqrt"],
'bootstrap' : ['True','False']
}

clf = GridSearchCV(model , params_rf , n_jobs=-1, cv= 5,verbose=1)

result = clf.fit(X_train, y_train)
best_params = result.best_params_

rfr = RandomForestRegressor(n_jobs=-1).set_params(**best_params)
rfr.fit(X_train,y_train)

# *********************** Final Result ********************

print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

print('Final Score is :', model_scores)